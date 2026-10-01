from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from sentinel.demo import report_candidates
from sentinel.store.db import make_engine
from sentinel.store.models import Account, Flag, Transaction

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def test_report_candidates_follow_queue_order_and_use_distinct_accounts(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'demo.db'}")
    with Session(engine) as session:
        for account_id in ("first", "second"):
            session.add(Account(id=account_id, created_at=NOW - timedelta(days=10)))
        session.flush()
        for index, account_id in enumerate(("first", "first", "second")):
            transaction = Transaction(
                id=f"tx_{index}",
                account_id=account_id,
                timestamp=NOW - timedelta(hours=index),
                amount=Decimal("50.00"),
                currency="USD",
                merchant_name="Corner Market",
                merchant_category="grocery",
                country="US",
                device_id="device_1",
                channel="online",
                memo=None,
            )
            session.add(transaction)
            session.flush()
            session.add(
                Flag(
                    transaction_id=transaction.id,
                    score=0.8,
                    reasons=["review"],
                    model_version="test",
                    created_at=NOW + timedelta(minutes=index),
                )
            )
        session.flush()
        assert [transaction.id for _, transaction in report_candidates(session, 2)] == [
            "tx_0",
            "tx_2",
        ]
        assert [transaction.id for _, transaction in report_candidates(session, 1)] == ["tx_0"]
        assert report_candidates(session, 0) == []
