"""Generate reproducible transactions and separate fraud labels."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TypedDict

import numpy as np

SPLITS = ("train", "val", "eval", "finetune")
START = datetime(2025, 1, 1, tzinfo=UTC)
MERCHANTS = (
    ("Corner Market", "grocery"),
    ("Fresh Basket", "grocery"),
    ("Metro Pharmacy", "pharmacy"),
    ("Northside Cafe", "dining"),
    ("Station Coffee", "dining"),
    ("City Transit", "transport"),
    ("Fuel Point", "fuel"),
    ("Home Supply", "retail"),
    ("Digital Books", "retail"),
    ("Bank Transfer", "transfer"),
)
COUNTRY_CURRENCIES = {
    "US": "USD",
    "GB": "GBP",
    "CA": "CAD",
    "DE": "EUR",
    "FR": "EUR",
    "AU": "AUD",
}
PATTERN_SIZES = {
    "velocity_burst": 6,
    "account_takeover": 4,
    "amount_spike": 3,
    "structuring": 4,
    "dormant_drain": 4,
}


class Transaction(TypedDict):
    transaction_id: str
    account_id: str
    timestamp: str
    amount: str
    currency: str
    merchant_name: str
    merchant_category: str
    country: str
    device_id: str
    channel: str
    memo: str | None


@dataclass(frozen=True)
class Account:
    account_id: str
    home_country: str
    currency: str
    usual_merchants: list[str]
    usual_categories: list[str]
    usual_devices: list[str]
    active_hours: list[int]
    lognormal_mu: float
    lognormal_sigma: float
    transactions_per_day: float


@dataclass
class Record:
    transaction: Transaction
    pattern: str | None = None
    episode_id: str | None = None


def _derived_seed(seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _money(value: float) -> str:
    amount = Decimal(str(max(0.01, value)))
    return str(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _accounts(rng: np.random.Generator, count: int, days: int, split: str) -> list[Account]:
    accounts = []
    base_frequency = max(1.1, 1200 / (count * days))
    countries = tuple(COUNTRY_CURRENCIES)
    for index in range(count):
        home_country = countries[int(rng.integers(len(countries)))]
        merchant_indices = rng.choice(len(MERCHANTS) - 1, size=4, replace=False)
        merchants = [MERCHANTS[int(i)] for i in merchant_indices]
        profile = int(rng.integers(3))
        hours = [*range(7, 19)] if profile == 0 else [*range(12, 24)]
        if profile == 2:
            hours = [*range(0, 7), *range(19, 24)]
        account_id = f"{split}-A{index + 1:06d}"
        accounts.append(
            Account(
                account_id=account_id,
                home_country=home_country,
                currency=COUNTRY_CURRENCIES[home_country],
                usual_merchants=[name for name, _ in merchants],
                usual_categories=sorted({category for _, category in merchants}),
                usual_devices=[f"{account_id}-D1", f"{account_id}-D2"],
                active_hours=hours,
                lognormal_mu=round(float(rng.uniform(math.log(18), math.log(90))), 6),
                lognormal_sigma=round(float(rng.uniform(0.35, 0.75)), 6),
                transactions_per_day=round(
                    float(base_frequency * rng.lognormal(0, 0.3)), 6
                ),
            )
        )
    return accounts


def _transaction(
    account: Account,
    timestamp: datetime,
    amount: float,
    merchant: tuple[str, str],
    country: str,
    device_id: str,
    channel: str,
    memo: str | None = None,
) -> Transaction:
    return {
        "transaction_id": "",
        "account_id": account.account_id,
        "timestamp": _timestamp(timestamp),
        "amount": _money(amount),
        "currency": account.currency,
        "merchant_name": merchant[0],
        "merchant_category": merchant[1],
        "country": country,
        "device_id": device_id,
        "channel": channel,
        "memo": memo,
    }


def _legitimate(rng: np.random.Generator, accounts: list[Account], days: int) -> list[Record]:
    records = []
    for account in accounts:
        merchant_lookup = [item for item in MERCHANTS if item[0] in account.usual_merchants]
        for day in range(days):
            count = int(rng.poisson(account.transactions_per_day))
            if day == 0:
                count = max(1, count)
            for _ in range(count):
                hour = int(rng.choice(account.active_hours))
                timestamp = START + timedelta(
                    days=day,
                    hours=hour,
                    minutes=int(rng.integers(60)),
                    seconds=int(rng.integers(60)),
                )
                is_transfer = bool(rng.random() < 0.04)
                merchant = (
                    ("Bank Transfer", "transfer")
                    if is_transfer
                    else merchant_lookup[int(rng.integers(len(merchant_lookup)))]
                )
                channel = (
                    "transfer"
                    if is_transfer
                    else ("online" if rng.random() < 0.35 else "card_present")
                )
                amount = float(rng.lognormal(account.lognormal_mu, account.lognormal_sigma))
                memo = "Monthly payment" if is_transfer and rng.random() < 0.15 else None
                records.append(
                    Record(
                        _transaction(
                            account,
                            timestamp,
                            amount,
                            merchant,
                            account.home_country,
                            str(rng.choice(account.usual_devices)),
                            channel,
                            memo,
                        )
                    )
                )
    return records


def _episode(
    rng: np.random.Generator,
    account: Account,
    pattern: str,
    episode_id: str,
    count: int,
    start: datetime,
    prior_amounts: list[float],
) -> list[Record]:
    records = []
    foreign_countries = [
        country for country in COUNTRY_CURRENCIES if country != account.home_country
    ]
    foreign_country = foreign_countries[int(rng.integers(len(foreign_countries)))]
    new_device = f"{account.account_id}-D3"
    p95 = float(np.percentile(prior_amounts, 95))
    for index in range(count):
        if pattern == "velocity_burst":
            timestamp = start + timedelta(seconds=index * 600 / max(1, count - 1))
            amount = float(rng.lognormal(account.lognormal_mu, account.lognormal_sigma))
            merchant = MERCHANTS[int(rng.integers(len(MERCHANTS) - 1))]
            country, device, channel = account.home_country, account.usual_devices[0], "online"
        elif pattern == "account_takeover":
            timestamp = start + timedelta(minutes=index * 4)
            amount = p95 * float(rng.uniform(1.2, 3.0))
            merchant = MERCHANTS[int(rng.integers(len(MERCHANTS) - 1))]
            country, device, channel = foreign_country, new_device, "online"
        elif pattern == "amount_spike":
            timestamp = start + timedelta(minutes=index * 3)
            amount = p95 * float(rng.uniform(5.0, 8.0))
            merchant = MERCHANTS[int(rng.integers(len(MERCHANTS) - 1))]
            country, device, channel = account.home_country, account.usual_devices[0], "online"
        elif pattern == "structuring":
            timestamp = start + timedelta(minutes=index * 2)
            amount = 9950.0
            merchant = ("Bank Transfer", "transfer")
            country, device, channel = account.home_country, account.usual_devices[0], "transfer"
        else:
            timestamp = start + timedelta(minutes=index * 3)
            amount = p95 * float(rng.uniform(6.0, 10.0))
            merchant = ("Bank Transfer", "transfer")
            country, device, channel = account.home_country, account.usual_devices[0], "transfer"
        records.append(
            Record(
                _transaction(account, timestamp, amount, merchant, country, device, channel),
                pattern,
                episode_id,
            )
        )
    return records


def generate_dataset(out: Path, seed: int, accounts: int, days: int, split: str) -> None:
    """Write one split's account, transaction, and label JSONL files."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if accounts < 5 or days < 14:
        raise ValueError("accounts must be at least 5 and days must be at least 14")
    rng = np.random.default_rng(_derived_seed(seed, split))
    account_rows = _accounts(rng, accounts, days, split)
    records = _legitimate(rng, account_rows, days)
    target_fraud = max(sum(PATTERN_SIZES.values()), round(len(records) * 0.0205))
    cycles = min(accounts // 5, max(1, round(target_fraud / sum(PATTERN_SIZES.values()))))
    scale = max(1.0, target_fraud / (cycles * sum(PATTERN_SIZES.values())))
    owner_order = rng.permutation(accounts)
    for episode_index in range(cycles * len(PATTERN_SIZES)):
        pattern = tuple(PATTERN_SIZES)[episode_index % len(PATTERN_SIZES)]
        account = account_rows[int(owner_order[episode_index])]
        count = max(PATTERN_SIZES[pattern], round(PATTERN_SIZES[pattern] * scale))
        if pattern == "dormant_drain":
            start = START + timedelta(days=days - 1, hours=12)
        else:
            start = START + timedelta(
                days=int(rng.integers(2, days - 2)),
                hours=int(rng.integers(8, 18)),
                minutes=int(rng.integers(60)),
            )
        prior_amounts = [
            float(record.transaction["amount"])
            for record in records
            if record.transaction["account_id"] == account.account_id
            and record.transaction["timestamp"] < _timestamp(start)
        ]
        if not prior_amounts:
            prior_amounts = [math.exp(account.lognormal_mu)]
        episode_id = f"{split}-E{episode_index + 1:06d}"
        episode = _episode(rng, account, pattern, episode_id, count, start, prior_amounts)
        end = episode[-1].transaction["timestamp"]
        quiet_start = _timestamp(start - timedelta(days=10)) if pattern == "dormant_drain" else None
        records = [
            record
            for record in records
            if not (
                record.transaction["account_id"] == account.account_id
                and record.pattern is None
                and (quiet_start or _timestamp(start))
                <= record.transaction["timestamp"]
                <= end
            )
        ]
        records.extend(episode)
    records.sort(
        key=lambda record: (record.transaction["timestamp"], record.transaction["account_id"])
    )
    for index, record in enumerate(records, start=1):
        record.transaction["transaction_id"] = f"{split}-T{index:08d}"
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "accounts.jsonl", [asdict(account) for account in account_rows])
    _write_jsonl(out / "transactions.jsonl", [record.transaction for record in records])
    _write_jsonl(
        out / "labels.jsonl",
        [
            {
                "transaction_id": record.transaction["transaction_id"],
                "is_fraud": record.pattern is not None,
                "pattern": record.pattern,
                "episode_id": record.episode_id,
            }
            for record in records
        ],
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--accounts", type=int, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()
    generate_dataset(args.out, args.seed, args.accounts, args.days, args.split)


if __name__ == "__main__":
    main()
