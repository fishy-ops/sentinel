"""Prepare a local demonstration database and serve the analyst console."""

import argparse
import os
from pathlib import Path

import httpx
import uvicorn
from sqlalchemy import select
from sqlalchemy.orm import Session

from data.synth import generate_dataset
from sentinel.agent.explainer import HttpChatClient, explain_flag
from sentinel.agent.tools import BoundTools
from sentinel.api.auth import create_key
from sentinel.detect.train import train
from sentinel.seed import seed
from sentinel.store.db import make_engine
from sentinel.store.models import Explanation, Flag, Transaction


def prepare(root: Path, explain_count: int = 0) -> tuple[str, str, int]:
    data_root = root / "data/generated"
    for split in ("train", "val", "eval", "finetune"):
        path = data_root / split
        if not all(
            (path / name).exists()
            for name in ("accounts.jsonl", "transactions.jsonl", "labels.jsonl")
        ):
            print(f"Generating {split} data…", flush=True)
            generate_dataset(path, seed=42, accounts=100, days=60, split=split)
    artifact = root / "models/iforest.joblib"
    if not artifact.exists() or not artifact.with_name("supervised.joblib").exists():
        print("Training detectors…", flush=True)
        train(data_root, artifact)
    database = root / "sentinel-demo.db"
    database.unlink(missing_ok=True)
    database_url = f"sqlite:///{database}"
    print("Seeding evaluation transactions and scoring flags…", flush=True)
    flag_count = seed(data_root / "eval", database_url, str(artifact), None)
    engine = make_engine(database_url)
    with Session(engine) as session:
        read_key = create_key(session, "demo analyst", {"read", "admin"})
        ingest_key = create_key(session, "demo ingest", {"ingest"})
    if explain_count:
        client = HttpChatClient()
        try:
            httpx.get(f"{client.base_url}/models", timeout=2).raise_for_status()
        except httpx.HTTPError:
            print("Report precomputation skipped: model endpoint is unreachable.", flush=True)
        else:
            with Session(engine) as session:
                seen: set[str] = set()
                flags = session.scalars(select(Flag).order_by(Flag.id)).all()
                for flag in flags:
                    transaction = session.get(Transaction, flag.transaction_id)
                    if transaction.account_id in seen:
                        continue
                    seen.add(transaction.account_id)
                    payload = explain_flag(BoundTools(session, flag, transaction), client)
                    session.add(
                        Explanation(
                            flag_id=flag.id,
                            model=payload["model"],
                            payload=payload,
                            created_at=flag.created_at,
                        )
                    )
                    session.commit()
                    if payload["failure"]:
                        print(f"Report for flag {flag.id}: {payload['failure']}", flush=True)
                    if len(seen) >= explain_count:
                        break
    print(f"Seeded {flag_count} flags.", flush=True)
    return read_key, ingest_key, flag_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain", type=int, default=0, help="precompute reports for N accounts")
    parser.add_argument("--no-serve", action="store_true", help="prepare the database and exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.explain < 0:
        parser.error("--explain must be nonnegative")
    root = Path.cwd()
    read_key, ingest_key, _ = prepare(root, args.explain)
    print(f"URL: http://{args.host}:{args.port}/", flush=True)
    print(f"Read + admin key: {read_key}", flush=True)
    print(f"Ingest key: {ingest_key}", flush=True)
    if not args.no_serve:
        os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{root / 'sentinel-demo.db'}"
        os.environ["SENTINEL_MODEL_ARTIFACT"] = str(root / "models/iforest.joblib")
        uvicorn.run("sentinel.api.main:app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
