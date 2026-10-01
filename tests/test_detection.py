import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from data.synth import generate_dataset
from sentinel.api.auth import create_key
from sentinel.api.main import create_app
from sentinel.api.settings import Settings
from sentinel.detect.features import FEATURE_NAMES, features_for_rows, transaction_features
from sentinel.detect.iforest import ForestModel
from sentinel.detect.rules import evaluate_rules

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def row(index: int, **changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "transaction_id": f"tx{index}",
        "account_id": "acct",
        "timestamp": (NOW - timedelta(days=10 - index)).isoformat(),
        "amount": "50.00",
        "currency": "USD",
        "merchant_name": "Shop",
        "merchant_category": "retail",
        "country": "US",
        "device_id": "device",
        "channel": "card_present",
        "memo": None,
    }
    result.update(changes)
    return result


def test_features_use_only_strictly_earlier_transactions() -> None:
    rows = [row(0), row(1), row(2)]
    before = features_for_rows(rows)
    future = row(3, amount="999999", country="GB", device_id="other")
    same_time = row(4, timestamp=rows[1]["timestamp"], amount="999999")
    after = features_for_rows([future, same_time, *rows])
    assert before[0] == after[2]
    assert before[1] == after[3]
    assert before[2] != after[4]


def test_offline_online_feature_parity() -> None:
    rows = [row(3), row(0), row(2), row(1)]
    offline = features_for_rows(pd.DataFrame(rows))
    for index, transaction in enumerate(rows):
        assert offline[index] == transaction_features(transaction, rows)
        assert set(offline[index]) == set(FEATURE_NAMES)


@pytest.mark.parametrize(
    ("name", "positive", "negative"),
    [
        ("velocity", {"account_age": 8, "count_10m": 3, "usual_max_10m": 2}, {"count_10m": 2}),
        (
            "account_takeover",
            {"account_age": 8, "is_new_device": 1, "is_new_country": 1, "amount_to_p95": 1},
            {"is_new_country": 0},
        ),
        (
            "amount_spike",
            {"account_age": 8, "amount_to_p95": 4, "amount": 400},
            {"amount_to_p95": 2},
        ),
        (
            "structuring",
            {
                "is_transfer": 1,
                "near_threshold": 1,
                "amount": 9500,
                "near_threshold_transfers_7d": 1,
            },
            {"near_threshold_transfers_7d": 0},
        ),
        (
            "dormant_drain",
            {"account_age": 8, "is_transfer": 1, "days_since_activity": 12, "amount_to_p95": 3},
            {"days_since_activity": 2},
        ),
    ],
)
def test_named_rules_positive_and_negative(
    name: str, positive: dict[str, float], negative: dict[str, float]
) -> None:
    base = dict.fromkeys(FEATURE_NAMES, 0.0)
    hits = evaluate_rules(base | positive)
    assert name in {hit.name for hit in hits}
    assert all(hit.reason for hit in hits)
    assert name not in {hit.name for hit in evaluate_rules(base | positive | negative)}


def test_cold_start_does_not_crash() -> None:
    features = transaction_features(row(0), [])
    assert features["account_age"] == 0
    assert evaluate_rules(features) == []


def test_rules_scorer_creates_api_flag(tmp_path: Path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'flags.db'}"), now=lambda: NOW)
    with Session(app.state.engine) as session:
        ingest = create_key(session, "ingest", {"ingest"})
        read = create_key(session, "read", {"read"})
    with TestClient(app) as client:
        for index in range(6):
            response = client.post(
                "/v1/transactions", headers={"X-API-Key": ingest}, json=row(index)
            )
            assert response.status_code == 201
        spike = row(7, amount="5000")
        response = client.post("/v1/transactions", headers={"X-API-Key": ingest}, json=spike)
        assert response.status_code == 201
        assert response.json()["flagged"] is True
        flags = client.get("/v1/flags", headers={"X-API-Key": read}).json()["items"]
        assert flags[0]["transaction_id"] == "tx7"
        assert "amount spike" in flags[0]["reasons"][0]
        assert flags[0]["model_version"] == "rules-1"


def test_train_cli_deterministic_for_seed(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for split in ("train", "val"):
        generate_dataset(root / split, seed=7, accounts=5, days=14, split=split)
    artifacts = [tmp_path / "first.joblib", tmp_path / "second.joblib"]
    for artifact in artifacts:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sentinel.detect.train",
                "--data",
                str(root),
                "--artifact",
                str(artifact),
                "--seed",
                "17",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    metadata = [json.loads(artifact.with_suffix(".json").read_text()) for artifact in artifacts]
    assert metadata[0] == metadata[1]
    models = [ForestModel.load(artifact) for artifact in artifacts]
    sample = transaction_features(row(0), [])
    assert (
        models[0].anomaly_scores([sample]).tolist() == models[1].anomaly_scores([sample]).tolist()
    )
    app = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'configured.db'}",
            model_artifact=str(artifacts[0]),
        )
    )
    assert app.state.scorer.forest.version == "iforest-1"
