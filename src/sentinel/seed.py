"""Load a generated data split into the database, scoring each transaction as it arrives."""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from sentinel.api.settings import Settings
from sentinel.detect.features import transaction_features
from sentinel.detect.score import CombinedScorer
from sentinel.store.db import make_engine
from sentinel.store.models import Account, Flag, Transaction


def seed(data: Path, database_url: str, artifact: str | None, accounts: int | None) -> int:
    rows = [json.loads(line) for line in (data / "transactions.jsonl").open()]
    if accounts is not None:
        keep = set(sorted({row["account_id"] for row in rows})[:accounts])
        rows = [row for row in rows if row["account_id"] in keep]
    scorer = CombinedScorer.from_artifact(artifact)
    history: dict[str, list[Transaction]] = defaultdict(list)
    transactions: list[Transaction] = []
    features: list[dict[str, float]] = []
    for row in rows:
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
        features.append(transaction_features(transaction, history[row["account_id"]]))
        history[row["account_id"]].append(transaction)
        transactions.append(transaction)
    forest_scores = scorer.forest.anomaly_scores(features) if scorer.forest and features else None
    supervised_scores = (
        scorer.supervised.probabilities(features) if scorer.supervised and features else None
    )
    flagged: list[Flag] = []
    for index, (transaction, feature) in enumerate(zip(transactions, features, strict=True)):
        result = scorer.detect(
            feature,
            forest_anomaly=float(forest_scores[index]) if forest_scores is not None else None,
            supervised_probability=(
                float(supervised_scores[index]) if supervised_scores is not None else None
            ),
        )
        if result.flagged:
            flagged.append(
                Flag(
                    transaction_id=transaction.id,
                    score=result.score,
                    reasons=result.reasons,
                    model_version=result.model_version,
                    created_at=datetime.now(UTC),
                )
            )
    with Session(make_engine(database_url)) as session:
        for account_id in sorted({row["account_id"] for row in rows}):
            session.add(Account(id=account_id, created_at=datetime.now(UTC)))
        session.add_all(transactions)
        session.flush()
        session.add_all(flagged)
        session.commit()
    return len(flagged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated/eval"))
    parser.add_argument("--artifact", default="models/iforest.joblib")
    parser.add_argument("--accounts", type=int, default=None, help="load only the first N accounts")
    args = parser.parse_args()
    settings = Settings()
    flags = seed(args.data, settings.database_url, args.artifact, args.accounts)
    print(f"loaded {args.data} into {settings.database_url}: {flags} flags")


if __name__ == "__main__":
    main()
