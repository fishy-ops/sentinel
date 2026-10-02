"""Build flagged-only typed decisions from fresh finetune data and existing val data."""

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from data.synth import generate_dataset
from sentinel.agent.tools import BoundTools
from sentinel.decide.digest import build_digest
from sentinel.decide.questions import PATTERNS, QUESTIONS
from sentinel.detect.data import read_jsonl
from sentinel.seed import seed
from sentinel.store.db import make_engine
from sentinel.store.models import Flag, Transaction


def split_examples(data: Path, artifact: str | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels = {row["transaction_id"]: row for row in read_jsonl(data / "labels.jsonl")}
    rows: list[dict[str, Any]] = []
    balance: Counter[str] = Counter()
    pattern_keys = {value: key for key, value in PATTERNS.items()}
    with tempfile.TemporaryDirectory(prefix="sentinel-decisions-") as directory:
        engine = make_engine(f"sqlite:///{Path(directory) / 'decisions.db'}")
        try:
            seed(data, str(engine.url), artifact, None)
            with Session(engine) as session:
                for flag in session.scalars(select(Flag).order_by(Flag.id)):
                    tx = session.get(Transaction, flag.transaction_id)
                    if tx is None:
                        raise ValueError("flag has no transaction")
                    label = labels[tx.id]
                    fraud = "yes" if label["is_fraud"] else "no"
                    pattern = pattern_keys[label["pattern"]] if label["is_fraud"] else "F"
                    digest = build_digest(BoundTools(session, flag, tx))
                    balance[f"fraud.{fraud}"] += 1
                    balance[f"pattern.{pattern}"] += 1
                    for question, answer in zip(QUESTIONS, (fraud, pattern), strict=True):
                        rows.append(
                            {
                                "digest": digest,
                                "question": question.name,
                                "allowed_labels": list(question.labels),
                                "answer": answer,
                                "transaction_id": tx.id,
                                "account_id": tx.account_id,
                            }
                        )
        finally:
            engine.dispose()
    return rows, dict(sorted(balance.items()))


def data_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build(
    out: Path = Path("training/data/decisions"),
    validation: Path = Path("data/generated/val"),
    artifact: str | None = "models/iforest.joblib",
    accounts: int = 600,
    seed_value: int = 1337,
    days: int = 60,
) -> dict[str, Any]:
    if validation.name != "val":
        raise ValueError("validation must be the val split, never eval")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-decision-data-") as directory:
        fresh = Path(directory) / "finetune"
        generate_dataset(fresh, seed=seed_value, accounts=accounts, days=days, split="finetune")
        sources = (("train", fresh, "finetune-"), ("valid", validation, "val-"))
        summary: dict[str, Any] = {"seed": seed_value, "accounts": accounts, "days": days}
        account_sets = []
        for name, source, prefix in sources:
            ids = {row["account_id"] for row in read_jsonl(source / "accounts.jsonl")}
            if not ids or any(not account.startswith(prefix) for account in ids):
                raise ValueError(f"{name} must use only {prefix} accounts")
            account_sets.append(ids)
            rows, balance = split_examples(source, artifact)
            if not rows:
                raise ValueError(f"{name} has no flagged transactions")
            with (out / f"{name}.jsonl").open("w", encoding="utf-8") as target:
                for row in rows:
                    target.write(json.dumps(row, sort_keys=True) + "\n")
            summary[name] = {"flags": len(rows) // 2, "examples": len(rows), "balance": balance}
            print(f"{name}: {json.dumps(summary[name], sort_keys=True)}", flush=True)
        if account_sets[0] & account_sets[1]:
            raise ValueError("training and validation accounts overlap")
        summary["data_hash"] = data_hash([out / "train.jsonl", out / "valid.jsonl"])
        (out / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("training/data/decisions"))
    parser.add_argument("--validation", type=Path, default=Path("data/generated/val"))
    parser.add_argument("--artifact", default="models/iforest.joblib")
    parser.add_argument("--accounts", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    build(args.out, args.validation, args.artifact, args.accounts, args.seed)


if __name__ == "__main__":
    main()
