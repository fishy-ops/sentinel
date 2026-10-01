"""Read-only tools for the analyst agent.

Every tool is bound to the account of the flagged transaction when the agent run starts.
No tool accepts an account or flag identifier, so the model cannot read another account.
Each returned record carries a `ref` that explanations must cite.
"""

import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.detect.features import REPORTING_THRESHOLD, transaction_features
from sentinel.store.models import Flag, Transaction

MAX_TEXT = 120
# The flagged transaction is always cited under this fixed ref; other transactions by their id.
FLAGGED_REF = "flagged_transaction"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Detector signals worth showing an analyst, with the wording the model sees.
SIGNALS = {
    "amount_to_p95": "flagged amount divided by the account's prior 95th percentile amount",
    "amount_to_median": "flagged amount divided by the account's prior median amount",
    "count_10m": "prior transactions on this account in the 10 minutes before",
    "count_1h": "prior transactions on this account in the hour before",
    "count_24h": "prior transactions on this account in the 24 hours before",
    "usual_max_10m": "most transactions this account ever made within 10 minutes before",
    "days_since_activity": "days since the account's previous transaction",
    "device_age_hours": "hours since this device was first seen on the account (0 = first use)",
    "country_age_hours": "hours since this country was first seen on the account (0 = first use)",
    "hour_deviation": "hours between this transaction's time of day and the account's usual time",
    "near_threshold_transfers_7d": "prior transfers of 9000 to 9999.99 in the last 7 days",
    "account_age": "number of prior transactions on the account",
}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return "".join(char for char in value if char.isprintable())[:MAX_TEXT]


def _round(value: float) -> float:
    return round(float(value), 2)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _transaction(row: Transaction, ref: str | None = None) -> dict[str, Any]:
    when = _utc(row.timestamp)
    return {
        "ref": ref or row.id,
        "transaction_id": row.id,
        "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weekday": when.strftime("%A"),
        "amount": _round(row.amount),
        "currency": row.currency,
        "country": row.country,
        "device_id": row.device_id,
        "channel": row.channel,
        "merchant_category": row.merchant_category,
        "untrusted_text": {"merchant_name": _clean(row.merchant_name), "memo": _clean(row.memo)},
    }


def _error(message: str) -> dict[str, Any]:
    return {"error": {"code": "invalid_arguments", "message": message}}


class BoundTools:
    def __init__(self, session: Session, flag: Flag, transaction: Transaction) -> None:
        self.session = session
        self.flag = flag
        self.transaction = transaction
        self.results: list[dict[str, Any]] = []

    def _history(self) -> list[Transaction]:
        """Transactions on the bound account strictly before the flagged one, newest first."""
        return list(
            self.session.scalars(
                select(Transaction)
                .where(
                    Transaction.account_id == self.transaction.account_id,
                    Transaction.timestamp < self.transaction.timestamp,
                )
                .order_by(Transaction.timestamp.desc(), Transaction.id.desc())
            )
        )

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None) if name in TOOL_NAMES else None
        if handler is None:
            return {"error": {"code": "unknown_tool", "message": "Unknown tool"}}
        if not isinstance(arguments, dict):
            return _error("Arguments must be a JSON object")
        allowed = set(TOOLS[name]["parameters"])
        if set(arguments) - allowed:
            return _error(
                "Unexpected argument. The account is fixed for this review and cannot be changed."
            )
        for key, value in arguments.items():
            low, high = TOOLS[name]["parameters"][key]
            if type(value) is not int or not low <= value <= high:
                return _error(f"{key} must be an integer from {low} to {high}")
        result = {"records": handler(**arguments)}
        self.results.append(result)
        return result

    def _tool_get_flag_detail(self) -> list[dict[str, Any]]:
        features = transaction_features(self.transaction, self._history())
        records: list[dict[str, Any]] = [
            _transaction(self.transaction, FLAGGED_REF),
            {"ref": "flag.score", "value": _round(self.flag.score), "meaning": "risk score 0-1"},
        ]
        for index, reason in enumerate(self.flag.reasons):
            records.append(
                {
                    "ref": f"flag.reason.{index}",
                    "value": reason,
                    "numbers": [float(n.replace(",", "")) for n in _NUMBER.findall(reason)],
                }
            )
        records += [
            {"ref": f"signal.{key}", "value": _round(features[key]), "meaning": meaning}
            for key, meaning in SIGNALS.items()
        ]
        records.append(
            {
                "ref": "policy.reporting_threshold",
                "value": REPORTING_THRESHOLD,
                "meaning": "transfers at or above this amount are reported to the regulator",
            }
        )
        return records

    def _tool_get_account_history(self, limit: int = 20) -> list[dict[str, Any]]:
        return [_transaction(row) for row in self._history()[:limit]]

    def _tool_get_account_stats(self) -> list[dict[str, Any]]:
        history = self._history()
        if not history:
            return [{"ref": "stats.transaction_count", "value": 0}]
        amounts = [float(row.amount) for row in history]
        first, last = _utc(history[-1].timestamp), _utc(history[0].timestamp)
        days = max(1.0, (last - first).total_seconds() / 86400)

        def common(values: list[Any], size: int) -> list[Any]:
            return [value for value, _ in Counter(values).most_common(size)]

        values: dict[str, Any] = {
            "transaction_count": len(history),
            "first_transaction": first.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "transactions_per_day": _round(len(history) / days),
            "median_amount": _round(median(amounts)),
            "p95_amount": _round(np.percentile(amounts, 95)),
            "max_amount": _round(max(amounts)),
            "countries_seen": sorted({row.country for row in history}),
            "devices_seen": sorted({row.device_id for row in history}),
            "usual_hours_utc": sorted(common([_utc(row.timestamp).hour for row in history], 6)),
            "transfer_count": sum(row.channel == "transfer" for row in history),
        }
        records: list[dict[str, Any]] = [
            {"ref": f"stats.{key}", "value": value} for key, value in values.items()
        ]
        records.append(
            {
                "ref": "stats.usual_merchants",
                "untrusted_text": {
                    "merchant_names": common([_clean(row.merchant_name) for row in history], 5)
                },
            }
        )
        return records

    def _tool_get_recent_activity(self, window_hours: int = 24) -> list[dict[str, Any]]:
        start = self.transaction.timestamp - timedelta(hours=window_hours)
        return [_transaction(row) for row in self._history() if row.timestamp >= start][:50]

    def _tool_get_prior_flags(self) -> list[dict[str, Any]]:
        rows = self.session.execute(
            select(Flag, Transaction)
            .join(Transaction, Flag.transaction_id == Transaction.id)
            .where(
                Transaction.account_id == self.transaction.account_id,
                Transaction.timestamp < self.transaction.timestamp,
            )
            .order_by(Transaction.timestamp.desc())
            .limit(20)
        ).all()
        return [
            {
                "ref": f"prior_flag.{flag.id}",
                "transaction": row.id,
                "timestamp": _utc(row.timestamp).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "amount": _round(row.amount),
                "score": _round(flag.score),
                "reasons": flag.reasons,
            }
            for flag, row in rows
        ]

    @property
    def refs(self) -> dict[str, dict[str, Any]]:
        return {record["ref"]: record for result in self.results for record in result["records"]}


TOOLS: dict[str, dict[str, Any]] = {
    "get_flag_detail": {
        "description": (
            "The flagged transaction, its risk score, the detector's reasons, and the signals "
            "behind them. Call this first."
        ),
        "parameters": {},
    },
    "get_account_stats": {
        "description": (
            "Summary of the account's behaviour before the flagged transaction: typical amounts, "
            "countries, devices, hours, and merchants. Use it to say what is normal."
        ),
        "parameters": {},
    },
    "get_recent_activity": {
        "description": (
            "Transactions on the account in the hours leading up to the flagged transaction."
        ),
        "parameters": {"window_hours": (1, 72)},
    },
    "get_account_history": {
        "description": "The account's most recent transactions before the flagged one.",
        "parameters": {"limit": (1, 50)},
    },
    "get_prior_flags": {
        "description": "Earlier flags raised on this account.",
        "parameters": {},
    },
}
TOOL_NAMES = tuple(TOOLS)
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "integer", "minimum": low, "maximum": high}
                    for key, (low, high) in spec["parameters"].items()
                },
                "additionalProperties": False,
            },
        },
    }
    for name, spec in TOOLS.items()
]
