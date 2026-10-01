from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleHit:
    name: str
    reason: str


def evaluate_rules(features: Mapping[str, float]) -> list[RuleHit]:
    f = features
    hits = []
    age = f["account_age"]
    if (
        age >= 5
        and f["count_10m"] >= 3
        and f["count_1h"] >= 3
        and f["count_24h"] >= 3
        and f["usual_max_10m"] <= 2
    ):
        hits.append(
            RuleHit(
                "velocity",
                f"velocity: {int(f['count_10m']) + 1} transactions in 10 minutes "
                f"(usual max {int(f['usual_max_10m'])})",
            )
        )
    recent_device = f["is_new_device"] or (
        f["device_seen_count"] > 0 and f["device_age_hours"] < 24
    )
    recent_country = f["is_new_country"] or (
        f["country_seen_count"] > 0 and f["country_age_hours"] < 24
    )
    support = sum(
        (
            f["hour_deviation"] >= 4,
            bool(f["is_new_merchant"]),
            bool(f["is_online"]),
            f["amount_to_median"] >= 1.5,
        )
    )
    if age >= 5 and recent_device and recent_country and support >= 2:
        hits.append(
            RuleHit(
                "account_takeover",
                "account takeover: device and country first seen within 24 hours; "
                f"{support} supporting signals, amount {f['amount_to_median']:.1f}x prior median",
            )
        )
    if age >= 5 and f["amount_to_p95"] >= 3 and f["amount"] >= 100:
        hits.append(
            RuleHit(
                "amount_spike",
                f"amount spike: {f['amount_to_p95']:.1f}x prior p95 on ${f['amount']:.2f}",
            )
        )
    if f["is_transfer"] and f["near_threshold"] and f["near_threshold_transfers_7d"] >= 1:
        hits.append(
            RuleHit(
                "structuring",
                f"structuring: ${f['amount']:.2f} transfer below $10000; "
                f"{int(f['near_threshold_transfers_7d'])} prior near-threshold transfers in 7 days",
            )
        )
    if age >= 5 and f["is_transfer"] and f["days_since_activity"] >= 10 and f["amount_to_p95"] >= 2:
        hits.append(
            RuleHit(
                "dormant_drain",
                f"dormant drain: transfer after {f['days_since_activity']:.1f} "
                f"inactive days at {f['amount_to_p95']:.1f}x prior p95",
            )
        )
    return hits
