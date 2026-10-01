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
    ("Harbor Foods", "grocery"),
    ("Green Grocer", "grocery"),
    ("Orchard Market", "grocery"),
    ("Metro Pharmacy", "pharmacy"),
    ("Wellness Pharmacy", "pharmacy"),
    ("Care Chemist", "pharmacy"),
    ("Northside Cafe", "dining"),
    ("Station Coffee", "dining"),
    ("River Bistro", "dining"),
    ("Noodle House", "dining"),
    ("Garden Bakery", "dining"),
    ("Central Deli", "dining"),
    ("City Transit", "transport"),
    ("Metro Rail", "transport"),
    ("Regional Taxi", "transport"),
    ("Fuel Point", "fuel"),
    ("Highway Fuel", "fuel"),
    ("Lake Gas", "fuel"),
    ("Home Supply", "retail"),
    ("Digital Books", "retail"),
    ("Oak Outfitters", "retail"),
    ("Market Electronics", "retail"),
    ("Family Shoes", "retail"),
    ("Toy Corner", "retail"),
    ("Pet Pantry", "retail"),
    ("Garden Tools", "retail"),
    ("West Hardware", "retail"),
    ("Cinema Tickets", "entertainment"),
    ("Museum Pass", "entertainment"),
    ("Fitness Club", "fitness"),
    ("Hair Studio", "services"),
    ("City Parking", "transport"),
    ("Cloud Storage", "subscription"),
    ("Phone Plan", "subscription"),
    ("Bank Transfer", "transfer"),
)
MEMOS = (
    "groceries",
    "weekend plans",
    "thank you",
    "monthly bill",
    "lunch with friends",
    "household items",
    "birthday gift",
    "appointment",
    "tickets",
    "supplies",
    "coffee run",
    "family trip",
    "subscription",
    "repair",
    "utilities",
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
    scenario: str | None = None


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
    base_frequency = max(1.1, 1400 / (count * days))
    countries = tuple(COUNTRY_CURRENCIES)
    for index in range(count):
        home_country = countries[int(rng.integers(len(countries)))]
        merchant_indices = rng.choice(len(MERCHANTS) - 1, size=6, replace=False)
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
                transactions_per_day=round(float(base_frequency * rng.lognormal(0, 0.3)), 6),
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
                is_rent = is_transfer and rng.random() < 0.12
                is_one_off = not is_transfer and rng.random() < 0.05
                merchant = (
                    ("Bank Transfer", "transfer")
                    if is_transfer
                    else (
                        (f"Local Shop {account.account_id}-{day}-{_}", "retail")
                        if is_one_off
                        else merchant_lookup[int(rng.integers(len(merchant_lookup)))]
                    )
                )
                channel = (
                    "transfer"
                    if is_transfer
                    else ("online" if rng.random() < 0.35 else "card_present")
                )
                amount = (
                    float(rng.uniform(1500, 9000))
                    if is_rent
                    else float(rng.lognormal(account.lognormal_mu, account.lognormal_sigma))
                )
                memo = (
                    "rent" if is_rent else str(rng.choice(MEMOS)) if rng.random() < 0.18 else None
                )
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


def _hard_negatives(
    rng: np.random.Generator, accounts: list[Account], records: list[Record], days: int
) -> None:
    owners = [accounts[int(index)] for index in rng.permutation(len(accounts))[:4]]
    travel, new_device, large_purchase, busy_day = owners
    travel_day = max(2, days // 3)
    foreign = [country for country in COUNTRY_CURRENCIES if country != travel.home_country]
    destination = str(rng.choice(foreign))
    for record in records:
        transaction = record.transaction
        when = datetime.fromisoformat(transaction["timestamp"])
        if (
            transaction["account_id"] == travel.account_id
            and travel_day <= (when - START).days < travel_day + 3
        ):
            transaction["country"] = destination
            record.scenario = "travel"
        if transaction["account_id"] == new_device.account_id and (when - START).days >= days // 2:
            if rng.random() < 0.7:
                transaction["device_id"] = f"{new_device.account_id}-D3"
                record.scenario = "new_device"
    for offset in range(3):
        timestamp = START + timedelta(days=travel_day + offset, hours=12)
        merchant = next(item for item in MERCHANTS if item[0] == travel.usual_merchants[offset])
        records.append(
            Record(
                _transaction(
                    travel,
                    timestamp,
                    float(rng.lognormal(travel.lognormal_mu, travel.lognormal_sigma)),
                    merchant,
                    destination,
                    travel.usual_devices[0],
                    "card_present",
                ),
                scenario="travel",
            )
        )
    records.append(
        Record(
            _transaction(
                new_device,
                START + timedelta(days=days // 2, hours=12),
                float(rng.lognormal(new_device.lognormal_mu, new_device.lognormal_sigma)),
                next(item for item in MERCHANTS if item[0] == new_device.usual_merchants[0]),
                new_device.home_country,
                f"{new_device.account_id}-D3",
                "online",
            ),
            scenario="new_device",
        )
    )
    prior = [
        float(record.transaction["amount"])
        for record in records
        if record.transaction["account_id"] == large_purchase.account_id
    ]
    p95 = float(np.percentile(prior, 95))
    records.append(
        Record(
            _transaction(
                large_purchase,
                START + timedelta(days=days // 2, hours=14),
                p95 * float(rng.uniform(3, 6)),
                ("Market Electronics", "retail"),
                large_purchase.home_country,
                large_purchase.usual_devices[0],
                "card_present",
            ),
            scenario="large_purchase",
        )
    )
    start = START + timedelta(days=days // 2, hours=15)
    interval = int(rng.integers(8, 15))
    for offset in range(int(rng.integers(5, 9))):
        merchant = str(rng.choice(busy_day.usual_merchants))
        category = next(category for name, category in MERCHANTS if name == merchant)
        records.append(
            Record(
                _transaction(
                    busy_day,
                    start + timedelta(minutes=offset * interval),
                    float(rng.lognormal(busy_day.lognormal_mu, busy_day.lognormal_sigma)),
                    (merchant, category),
                    busy_day.home_country,
                    busy_day.usual_devices[0],
                    "card_present",
                ),
                scenario="busy_day",
            )
        )


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
    new_device = f"{account.account_id}-D{int(rng.integers(4, 8))}"
    p95 = float(np.percentile(prior_amounts, 95))
    elapsed = timedelta()
    for index in range(count):
        if pattern == "velocity_burst":
            if index:
                elapsed += timedelta(seconds=float(rng.uniform(5, 90)))
            amount = (
                float(rng.uniform(0.5, 6))
                if rng.random() < 0.35
                else float(rng.lognormal(account.lognormal_mu, account.lognormal_sigma))
            )
            merchant = MERCHANTS[int(rng.integers(len(MERCHANTS) - 1))]
            country, device, channel = (
                account.home_country,
                str(rng.choice(account.usual_devices)),
                "online",
            )
        elif pattern == "account_takeover":
            if index:
                elapsed += timedelta(minutes=int(rng.integers(7, 60)))
            amount = p95 * float(rng.uniform(0.4, 1.5))
            merchant = MERCHANTS[int(rng.integers(len(MERCHANTS) - 1))]
            country = account.home_country if rng.random() < 0.5 else foreign_country
            device, channel = new_device, "online"
        elif pattern == "amount_spike":
            if index:
                elapsed += timedelta(minutes=int(rng.integers(5, 90)))
            amount = p95 * float(rng.uniform(2.5, 8.0))
            merchant = MERCHANTS[int(rng.integers(len(MERCHANTS) - 1))]
            country, device, channel = (
                account.home_country,
                str(rng.choice(account.usual_devices)),
                "online",
            )
        elif pattern == "structuring":
            if index:
                elapsed += timedelta(hours=int(rng.integers(4, 13)))
            amount = float(rng.uniform(9000, 9990))
            merchant = ("Bank Transfer", "transfer")
            country, device, channel = (
                account.home_country,
                str(rng.choice(account.usual_devices)),
                "transfer",
            )
        else:
            if index:
                elapsed += timedelta(minutes=int(rng.integers(10, 90)))
            amount = p95 * float(rng.uniform(3.0, 9.0))
            merchant = ("Bank Transfer", "transfer")
            country, device, channel = (
                account.home_country,
                str(rng.choice(account.usual_devices)),
                "transfer",
            )
        records.append(
            Record(
                _transaction(account, start + elapsed, amount, merchant, country, device, channel),
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
    _hard_negatives(rng, account_rows, records, days)
    target_fraud = max(sum(PATTERN_SIZES.values()), round(len(records) * 0.0205))
    cycles = min(accounts // 5, max(1, int(target_fraud / sum(PATTERN_SIZES.values()))))
    scale = max(1.0, target_fraud / (cycles * sum(PATTERN_SIZES.values())))
    owner_order = rng.permutation(accounts)
    hard_negative_owners = {
        record.transaction["account_id"] for record in records if record.scenario is not None
    }
    dormant_positions = set(range(4, cycles * len(PATTERN_SIZES), len(PATTERN_SIZES)))
    for position in sorted(dormant_positions):
        if account_rows[int(owner_order[position])].account_id in hard_negative_owners:
            replacement = next(
                index
                for index in range(accounts)
                if index not in dormant_positions
                and account_rows[int(owner_order[index])].account_id not in hard_negative_owners
            )
            owner_order[position], owner_order[replacement] = (
                owner_order[replacement],
                owner_order[position],
            )
    for episode_index in range(cycles * len(PATTERN_SIZES)):
        pattern = tuple(PATTERN_SIZES)[episode_index % len(PATTERN_SIZES)]
        account = account_rows[int(owner_order[episode_index])]
        count = max(PATTERN_SIZES[pattern], round(PATTERN_SIZES[pattern] * scale))
        if pattern == "velocity_burst":
            count = int(rng.integers(5, 13))
        if pattern == "dormant_drain":
            quiet_days = int(rng.integers(10, min(30, days - 2) + 1))
            start = START + timedelta(days=int(rng.integers(quiet_days + 1, days)), hours=12)
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
        if pattern == "dormant_drain":
            quiet_start = _timestamp(start - timedelta(days=quiet_days))
            records = [
                record
                for record in records
                if not (
                    record.transaction["account_id"] == account.account_id
                    and record.pattern is None
                    and quiet_start <= record.transaction["timestamp"] < _timestamp(start)
                )
            ]
            records.append(
                Record(
                    _transaction(
                        account,
                        start - timedelta(days=quiet_days),
                        math.exp(account.lognormal_mu),
                        next(item for item in MERCHANTS if item[0] == account.usual_merchants[0]),
                        account.home_country,
                        account.usual_devices[0],
                        "card_present",
                    )
                )
            )
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
                "scenario": record.scenario,
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
