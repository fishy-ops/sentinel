import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from statistics import median, pstdev
from typing import Any

import numpy as np
from pandas import DataFrame

FEATURE_NAMES = (
    "amount",
    "account_age",
    "amount_zscore",
    "amount_to_p95",
    "amount_to_median",
    "count_10m",
    "count_1h",
    "count_24h",
    "sum_10m",
    "sum_1h",
    "sum_24h",
    "seconds_since_previous",
    "days_since_activity",
    "is_new_device",
    "is_new_country",
    "is_new_merchant",
    "device_seen_count",
    "country_seen_count",
    "hour_deviation",
    "is_transfer",
    "near_threshold",
    "near_threshold_transfers_7d",
    "usual_max_10m",
)


def _field(row: Mapping[str, Any] | object, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, "id" if name == "transaction_id" else name)


def _time(row: Mapping[str, Any] | object) -> datetime:
    value = _field(row, "timestamp")
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def transaction_features(
    transaction: Mapping[str, Any] | object,
    history: Iterable[Mapping[str, Any] | object],
) -> dict[str, float]:
    when = _time(transaction)
    account = _field(transaction, "account_id")
    prior = [row for row in history if _field(row, "account_id") == account and _time(row) < when]
    prior.sort(key=_time)
    amount = float(_field(transaction, "amount"))
    amounts = [float(_field(row, "amount")) for row in prior]
    age = len(prior)
    p95 = float(np.percentile(amounts, 95)) if amounts else 0.0
    med = float(median(amounts)) if amounts else 0.0
    mean = sum(amounts) / age if age else 0.0
    std = pstdev(amounts) if age > 1 else 0.0
    windows = {
        name: [row for row in prior if when - _time(row) <= delta]
        for name, delta in (
            ("10m", timedelta(minutes=10)),
            ("1h", timedelta(hours=1)),
            ("24h", timedelta(days=1)),
        )
    }
    previous = _time(prior[-1]) if prior else None
    device = _field(transaction, "device_id")
    country = _field(transaction, "country")
    merchant = _field(transaction, "merchant_name")
    device_count = sum(_field(row, "device_id") == device for row in prior)
    country_count = sum(_field(row, "country") == country for row in prior)
    merchant_count = sum(_field(row, "merchant_name") == merchant for row in prior)
    hours = [_time(row).hour + _time(row).minute / 60 for row in prior]
    if hours:
        angle = math.atan2(
            sum(math.sin(2 * math.pi * hour / 24) for hour in hours),
            sum(math.cos(2 * math.pi * hour / 24) for hour in hours),
        )
        usual_hour = (angle * 12 / math.pi) % 24
        difference = abs(when.hour + when.minute / 60 - usual_hour)
        hour_deviation = min(difference, 24 - difference)
    else:
        hour_deviation = 0.0
    usual_max = 0
    prior_times = [_time(row) for row in prior]
    left = 0
    for right, point in enumerate(prior_times):
        while point - prior_times[left] > timedelta(minutes=10):
            left += 1
        usual_max = max(usual_max, right - left + 1)
    return {
        "amount": amount,
        "account_age": float(age),
        "amount_zscore": (amount - mean) / std if std > 0 else 0.0,
        "amount_to_p95": amount / p95 if p95 > 0 else 0.0,
        "amount_to_median": amount / med if med > 0 else 0.0,
        "count_10m": float(len(windows["10m"])),
        "count_1h": float(len(windows["1h"])),
        "count_24h": float(len(windows["24h"])),
        "sum_10m": sum(float(_field(row, "amount")) for row in windows["10m"]),
        "sum_1h": sum(float(_field(row, "amount")) for row in windows["1h"]),
        "sum_24h": sum(float(_field(row, "amount")) for row in windows["24h"]),
        "seconds_since_previous": (when - previous).total_seconds() if previous else 0.0,
        "days_since_activity": (when - previous).total_seconds() / 86400 if previous else 0.0,
        "is_new_device": float(device_count == 0),
        "is_new_country": float(country_count == 0),
        "is_new_merchant": float(merchant_count == 0),
        "device_seen_count": float(device_count),
        "country_seen_count": float(country_count),
        "hour_deviation": hour_deviation,
        "is_transfer": float(_field(transaction, "channel") == "transfer"),
        "near_threshold": float(9000 <= amount < 10000),
        "near_threshold_transfers_7d": float(
            sum(
                _field(row, "channel") == "transfer"
                and 9000 <= float(_field(row, "amount")) < 10000
                and when - _time(row) <= timedelta(days=7)
                for row in prior
            )
        ),
        "usual_max_10m": float(usual_max),
    }


def features_for_rows(rows: Iterable[Mapping[str, Any]] | DataFrame) -> list[dict[str, float]]:
    records = rows.to_dict("records") if isinstance(rows, DataFrame) else list(rows)
    order = sorted(
        range(len(records)),
        key=lambda index: (_time(records[index]), str(_field(records[index], "transaction_id"))),
    )
    history: dict[str, list[Mapping[str, Any]]] = {}
    result: list[dict[str, float]] = [{} for _ in records]
    for index in order:
        row = records[index]
        account = str(_field(row, "account_id"))
        prior = history.setdefault(account, [])
        result[index] = transaction_features(row, prior)
        prior.append(row)
    return result
