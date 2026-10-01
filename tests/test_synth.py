import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from data.synth import PATTERN_SIZES, SPLITS, generate_dataset


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def splits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("splits")
    paths = {split: root / split for split in SPLITS}
    for split, path in paths.items():
        generate_dataset(path, seed=47, accounts=12, days=20, split=split)
    return paths


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate_dataset(first, seed=11, accounts=8, days=14, split="train")
    generate_dataset(second, seed=11, accounts=8, days=14, split="train")
    for name in ("accounts.jsonl", "transactions.jsonl", "labels.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_different_seeds_differ(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate_dataset(first, seed=11, accounts=8, days=14, split="train")
    generate_dataset(second, seed=12, accounts=8, days=14, split="train")
    assert (first / "transactions.jsonl").read_bytes() != (
        second / "transactions.jsonl"
    ).read_bytes()


def test_splits_have_disjoint_accounts(splits: dict[str, Path]) -> None:
    seen: set[str] = set()
    for split, path in splits.items():
        ids = {str(row["account_id"]) for row in _read(path / "accounts.jsonl")}
        assert all(account_id.startswith(f"{split}-") for account_id in ids)
        assert seen.isdisjoint(ids)
        seen.update(ids)


@pytest.mark.parametrize("split", SPLITS)
def test_patterns_rate_and_record_shape(splits: dict[str, Path], split: str) -> None:
    path = splits[split]
    transactions = _read(path / "transactions.jsonl")
    labels = _read(path / "labels.jsonl")
    assert len(transactions) == len(labels)
    assert 0.01 <= sum(bool(row["is_fraud"]) for row in labels) / len(labels) <= 0.03
    assert {row["pattern"] for row in labels if row["is_fraud"]} == set(PATTERN_SIZES)
    assert [row["transaction_id"] for row in transactions] == [
        row["transaction_id"] for row in labels
    ]
    assert all("is_fraud" not in row and "pattern" not in row and "episode_id" not in row
               for row in transactions)
    timestamps = [datetime.fromisoformat(str(row["timestamp"])) for row in transactions]
    assert timestamps == sorted(timestamps)
    assert all(timestamp.utcoffset().total_seconds() == 0 for timestamp in timestamps)
    for row in transactions:
        amount = str(row["amount"])
        assert Decimal(amount) > 0
        assert amount == f"{Decimal(amount):.2f}"


@pytest.mark.parametrize("split", SPLITS)
def test_episodes_have_expected_behavior(splits: dict[str, Path], split: str) -> None:
    transactions = _read(splits[split] / "transactions.jsonl")
    labels = _read(splits[split] / "labels.jsonl")
    by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for transaction, label in zip(transactions, labels, strict=True):
        if label["is_fraud"]:
            by_episode[str(label["episode_id"])].append(transaction)
    patterns = {str(label["episode_id"]): str(label["pattern"]) for label in labels}
    for episode_id, rows in by_episode.items():
        pattern = patterns[episode_id]
        assert len(rows) >= PATTERN_SIZES[pattern]
        account_id = rows[0]["account_id"]
        assert all(row["account_id"] == account_id for row in rows)
        times = [datetime.fromisoformat(str(row["timestamp"])) for row in rows]
        same_account_between = [
            row for row in transactions
            if row["account_id"] == account_id
            and str(rows[0]["timestamp"]) <= str(row["timestamp"]) <= str(rows[-1]["timestamp"])
        ]
        assert same_account_between == rows
        if pattern == "structuring":
            amounts = {Decimal(str(row["amount"])) for row in rows}
            assert len(amounts) == 1
            assert all(9000 <= amount < 10000 for amount in amounts)
        elif pattern == "velocity_burst":
            assert (max(times) - min(times)).total_seconds() <= 15 * 60
        elif pattern == "amount_spike":
            prior_amounts = [
                float(str(row["amount"]))
                for row in transactions
                if row["account_id"] == account_id
                and str(row["timestamp"]) < str(rows[0]["timestamp"])
            ]
            assert prior_amounts
            assert min(Decimal(str(row["amount"])) for row in rows) > Decimal(
                str(5 * np.percentile(prior_amounts, 95) - 0.01)
            )
        elif pattern == "account_takeover":
            earlier = [
                row for row in transactions
                if row["account_id"] == account_id
                and str(row["timestamp"]) < str(rows[0]["timestamp"])
            ]
            assert earlier
            old_devices = {row["device_id"] for row in earlier}
            old_countries = {row["country"] for row in earlier}
            assert all(row["device_id"] not in old_devices for row in rows)
            assert all(row["country"] not in old_countries for row in rows)
        elif pattern == "dormant_drain":
            earlier_times = [
                datetime.fromisoformat(str(row["timestamp"]))
                for row in transactions
                if row["account_id"] == account_id
                and str(row["timestamp"]) < str(rows[0]["timestamp"])
            ]
            assert earlier_times
            assert (min(times) - max(earlier_times)).days >= 10


def test_invalid_size_and_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="accounts"):
        generate_dataset(tmp_path, seed=1, accounts=4, days=14, split="train")
    with pytest.raises(ValueError, match="split"):
        generate_dataset(tmp_path, seed=1, accounts=5, days=14, split="other")


def test_minimum_supported_size_has_target_fraud_rate(tmp_path: Path) -> None:
    generate_dataset(tmp_path, seed=3, accounts=5, days=14, split="eval")
    labels = _read(tmp_path / "labels.jsonl")
    rate = sum(bool(row["is_fraud"]) for row in labels) / len(labels)
    assert 0.01 <= rate <= 0.03
