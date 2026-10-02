"""Compact evidence from the same time-bounded records used for flag review."""

import json
from typing import Any

from sentinel.agent.tools import BoundTools

MAX_CHARS = 1250
MAX_TOKENS = 350


def untrusted_line(key: str, value: str | None) -> str:
    text = "".join(char for char in (value or "") if char.isprintable())[:60]
    quoted = json.dumps(text, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"untrusted.{key}: {quoted}"


def build_digest(tools: BoundTools) -> str:
    detail = tools.call("get_flag_detail", {})["records"]
    stats = tools.call("get_account_stats", {})["records"]
    tx = detail[0]
    values = {row["ref"]: row.get("value") for row in (*detail[1:], *stats)}
    lines = [
        f"amount: {tx['amount']} {tx['currency']}",
        f"time_utc: {tx['timestamp']}",
        f"country: {tx['country']}",
        f"channel: {tx['channel']}",
        f"category: {tx['merchant_category']}",
        untrusted_line("merchant", tx["untrusted_text"]["merchant_name"]),
        untrusted_line("memo", tx["untrusted_text"]["memo"]),
    ]
    fields = (
        ("prior_count", "stats.transaction_count"),
        ("prior_median", "stats.median_amount"),
        ("prior_p95", "stats.p95_amount"),
        ("amount_over_p95", "signal.amount_to_p95"),
        ("prior_10m", "signal.count_10m"),
        ("prior_1h", "signal.count_1h"),
        ("prior_24h", "signal.count_24h"),
        ("usual_max_10m", "signal.usual_max_10m"),
        ("device_age_h", "signal.device_age_hours"),
        ("country_age_h", "signal.country_age_hours"),
        ("inactive_days", "signal.days_since_activity"),
        ("near_threshold_7d", "signal.near_threshold_transfers_7d"),
        ("reporting_threshold", "policy.reporting_threshold"),
        ("hour_deviation", "signal.hour_deviation"),
        ("prior_countries", "stats.countries_seen"),
        ("prior_per_day", "stats.transactions_per_day"),
        ("prior_max", "stats.max_amount"),
        ("usual_hours", "stats.usual_hours_utc"),
        ("score", "flag.score"),
    )
    for key, ref in fields:
        if ref in values:
            value = json.dumps(values[ref], ensure_ascii=False, separators=(",", ":"))
            lines.append(f"{key}: {value}")
    for row in detail:
        if row["ref"].startswith("flag.reason."):
            reason = "".join(char for char in row["value"] if char.isprintable())[:100]
            lines.append(f"reason: {reason}")
    # Keep complete lines, dropping the least essential trailing fields first.
    while len("\n".join(lines)) > MAX_CHARS:
        lines.pop()
    return "\n".join(lines)


def token_bounded(digest: str, tokenizer: Any) -> str:
    """The actual tokenizer enforces the budget at both training and inference."""
    lines = digest.splitlines()
    while len(tokenizer.encode("\n".join(lines), add_special_tokens=False)) > MAX_TOKENS:
        if len(lines) <= 7:
            raise ValueError("core digest exceeds the 350-token budget")
        lines.pop()
    return "\n".join(lines)
