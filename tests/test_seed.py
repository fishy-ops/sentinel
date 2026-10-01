from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data.synth import generate_dataset
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
