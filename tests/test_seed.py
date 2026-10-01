import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data.synth import generate_dataset
from sentinel.detect.score import CombinedScorer
from sentinel.seed import seed
from sentinel.store.db import make_engine
from sentinel.store.models import Account, Flag, Transaction


def test_seed_small_generated_split_and_repeat_is_atomic(tmp_path: Path) -> None:
    data = tmp_path / "eval"
    generate_dataset(data, seed=7, accounts=5, days=14, split="eval")
    transaction_count = len((data / "transactions.jsonl").read_text().splitlines())
    db = f"sqlite:///{tmp_path / 'seed.db'}"
    flags = seed(data, db, None, None)
    assert flags > 0
    engine = make_engine(db)
    with Session(engine) as session:
        before = tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (Account, Transaction, Flag)
        )
    assert before == (5, transaction_count, flags)
    with pytest.raises(IntegrityError):
        seed(data, db, None, None)
    with Session(engine) as session:
        after = tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (Account, Transaction, Flag)
        )
    assert after == before


def test_batched_seed_matches_serial_historical_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "eval"
    generate_dataset(data, seed=9, accounts=5, days=14, split="eval")

    class Forest:
        threshold = 0.55
        version = "forest-test"

        def anomaly_scores(self, features: list[dict[str, float]]) -> list[float]:
            return [min(1.0, row["amount"] / 300) for row in features]

        def top_features(self, feature: dict[str, float]) -> list[str]:
            return [f"amount {feature['amount']:.2f}"]

    class Supervised:
        threshold = 0.7
        version = "supervised-test"

        def probabilities(self, features: list[dict[str, float]]) -> list[float]:
            return [min(1.0, row["count_24h"] / 10) for row in features]

        def top_features(self, feature: dict[str, float]) -> list[str]:
            return [f"count {feature['count_24h']:.2f}"]

    scorer = CombinedScorer(Forest(), Supervised())
    history: dict[str, list[Transaction]] = defaultdict(list)
    expected: dict[str, tuple[float, list[str], str]] = {}
    for line in (data / "transactions.jsonl").read_text().splitlines():
        row = json.loads(line)
        transaction = Transaction(
            id=row["transaction_id"],
            account_id=row["account_id"],
            timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
            amount=Decimal(row["amount"]),
            currency=row["currency"],
            merchant_name=row["merchant_name"],
            merchant_category=row["merchant_category"],
            country=row["country"],
            device_id=row["device_id"],
            channel=row["channel"],
            memo=row["memo"],
        )
        flag = scorer.score(transaction, history[transaction.account_id])
        if flag:
            expected[transaction.id] = (flag.score, flag.reasons, flag.model_version)
        history[transaction.account_id].append(transaction)
    monkeypatch.setattr(CombinedScorer, "from_artifact", classmethod(lambda cls, artifact: scorer))
    db = f"sqlite:///{tmp_path / 'batched.db'}"
    assert seed(data, db, "fake-artifact", None) == len(expected)
    with Session(make_engine(db)) as session:
        actual = {
            flag.transaction_id: (flag.score, flag.reasons, flag.model_version)
            for flag in session.scalars(select(Flag))
        }
    assert actual == expected
