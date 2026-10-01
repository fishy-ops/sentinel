import os
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.api.auth import create_key
from sentinel.api.main import create_app
from sentinel.api.settings import Settings
from sentinel.audit.chain import verify
from sentinel.store.models import ApiKey, AuditLog, Flag, Transaction

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class Fixture:
    def __init__(self, path: Path, burst: int = 100, max_body_bytes: int = 4096) -> None:
        self.time = 0.0
        self.app = create_app(
            Settings(
                database_url=f"sqlite:///{path}",
                rate_per_second=1,
                rate_burst=burst,
                max_body_bytes=max_body_bytes,
            ),
            clock=lambda: self.time,
            now=lambda: NOW,
        )
        self.client = TestClient(self.app)
        with Session(self.app.state.engine) as session:
            self.ingest = create_key(session, "ingest", {"ingest"})
            self.read = create_key(session, "read", {"read"})
            self.admin = create_key(session, "admin", {"admin"})

    def headers(self, key: str | None = None) -> dict[str, str]:
        return {"X-API-Key": key or self.ingest}


@pytest.fixture
def api(tmp_path: Path) -> Iterator[Fixture]:
    fixture = Fixture(tmp_path / "api.db")
    yield fixture
    fixture.client.close()


def transaction(transaction_id: str = "tx_1", **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "transaction_id": transaction_id,
        "account_id": "acct_1",
        "timestamp": NOW.isoformat(),
        "amount": "25.25",
        "currency": "USD",
        "merchant_name": "Corner Market",
        "merchant_category": "grocery",
        "country": "US",
        "device_id": "device_1",
        "channel": "card_present",
        "memo": "groceries",
    }
    row.update(changes)
    return row


def test_auth_scope_and_revocation(api: Fixture) -> None:
    client = api.client
    assert client.get("/v1/healthz").status_code == 200
    for headers in ({}, {"X-API-Key": "sk_ffffffff_bad"}):
        response = client.get("/v1/flags", headers=headers)
        assert response.status_code == 401
        assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]
    assert client.get("/v1/flags", headers=api.headers()).status_code == 403
    with Session(api.app.state.engine) as session:
        key = session.scalar(select(ApiKey).where(ApiKey.prefix == api.read.split("_")[1]))
        assert key is not None
        key.revoked = True
        session.commit()
    assert client.get("/v1/flags", headers=api.headers(api.read)).status_code == 401


def test_unknown_prefix_uses_dummy_hash(api: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import sentinel.api.auth as auth

    calls = []
    original = auth.hmac.compare_digest

    def compare(first: str, second: str) -> bool:
        calls.append((first, second))
        return original(first, second)

    monkeypatch.setattr(auth.hmac, "compare_digest", compare)
    assert (
        api.client.get("/v1/flags", headers={"X-API-Key": "sk_ffffffff_secret"}).status_code == 401
    )
    assert len(calls) == 1
    assert len(calls[0][0]) == len(calls[0][1]) == 64


def test_rate_limit_and_recovery(tmp_path: Path) -> None:
    api = Fixture(tmp_path / "rate.db", burst=2)
    for _ in range(2):
        assert api.client.get("/v1/flags", headers=api.headers(api.read)).status_code == 200
    limited = api.client.get("/v1/flags", headers=api.headers(api.read))
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "1"
    assert limited.headers["X-RateLimit-Remaining"] == "0"
    api.time += 1
    assert api.client.get("/v1/flags", headers=api.headers(api.read)).status_code == 200
    for _ in range(3):
        failure = api.client.get("/v1/flags")
    assert failure.status_code == 429


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"transaction_id": "bad id"}, "transaction_id"),
        ({"account_id": "!"}, "account_id"),
        ({"device_id": "!"}, "device_id"),
        ({"amount": 12.5}, "amount"),
        ({"amount": "0"}, "amount"),
        ({"amount": "1000000.01"}, "amount"),
        ({"amount": "1.001"}, "amount"),
        ({"currency": "ZZZ"}, "currency"),
        ({"country": "zz"}, "country"),
        ({"country": "ZZ"}, "country"),
        ({"timestamp": NOW.replace(tzinfo=None).isoformat()}, "timestamp"),
        ({"timestamp": (NOW + timedelta(minutes=6)).isoformat()}, "timestamp"),
        ({"timestamp": (NOW - timedelta(days=365 * 6)).isoformat()}, "timestamp"),
        ({"merchant_name": ""}, "merchant_name"),
        ({"merchant_name": "X" * 121}, "merchant_name"),
        ({"merchant_name": "secret\ntext"}, "merchant_name"),
        ({"memo": "X" * 281}, "memo"),
        ({"memo": "secret\x00text"}, "memo"),
        ({"channel": "cash"}, "channel"),
        ({"unknown": "secret"}, "unknown"),
    ],
)
def test_validation(api: Fixture, change: dict[str, object], field: str) -> None:
    response = api.client.post(
        "/v1/transactions", headers=api.headers(), json=transaction(**change)
    )
    assert response.status_code == 422
    assert field in response.json()["error"]["message"]
    assert "secret" not in response.text


def test_body_duplicate_idempotency_and_batch(api: Fixture) -> None:
    headers = api.headers() | {"Idempotency-Key": "once"}
    first = api.client.post("/v1/transactions", headers=headers, json=transaction())
    assert first.status_code == 201
    assert (
        api.client.post("/v1/transactions", headers=headers, json=transaction()).json()
        == first.json()
    )
    assert (
        api.client.post(
            "/v1/transactions", headers=headers, json=transaction(amount="27")
        ).status_code
        == 422
    )
    assert (
        api.client.post("/v1/transactions", headers=api.headers(), json=transaction()).status_code
        == 409
    )
    assert (
        api.client.post("/v1/transactions", headers=api.headers(), content=b"x" * 5000).status_code
        == 413
    )
    assert (
        api.client.post(
            "/v1/transactions/batch",
            headers=api.headers(),
            json=[transaction("a"), transaction("b")],
        ).status_code
        == 201
    )
    assert (
        api.client.post(
            "/v1/transactions/batch", headers=api.headers(), json=[transaction("c")] * 2
        ).status_code
        == 409
    )


def test_batch_limit(tmp_path: Path) -> None:
    api = Fixture(tmp_path / "batch.db", max_body_bytes=1_000_000)
    response = api.client.post(
        "/v1/transactions/batch",
        headers=api.headers(),
        json=[transaction(str(i)) for i in range(501)],
    )
    assert response.status_code == 422


def test_history_flags_pagination_and_audit(api: Fixture) -> None:
    for index in range(3):
        row = transaction(f"tx_{index}", timestamp=(NOW - timedelta(hours=index)).isoformat())
        assert (
            api.client.post("/v1/transactions", headers=api.headers(), json=row).status_code == 201
        )
    page = api.client.get(
        "/v1/accounts/acct_1/history?limit=2&offset=1", headers=api.headers(api.read)
    )
    assert [row["transaction_id"] for row in page.json()["items"]] == ["tx_1", "tx_2"]
    assert (
        api.client.get(
            "/v1/accounts/acct_1/history?limit=201", headers=api.headers(api.read)
        ).status_code
        == 422
    )
    assert (
        api.client.get("/v1/flags?account_id=acct_1", headers=api.headers(api.read)).json()["items"]
        == []
    )
    assert api.client.get("/v1/flags/99", headers=api.headers(api.read)).status_code == 404
    checked = api.client.get("/v1/audit/verify", headers=api.headers(api.admin))
    assert checked.json() == {"ok": True, "first_broken_entry_id": None}
    with Session(api.app.state.engine) as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.sequence)).all()
        assert len(rows) == 8
        assert all(row.entry_hash and row.prev_hash for row in rows)
        audit_text = " ".join(str(row.__dict__) for row in rows)
        assert api.ingest not in audit_text
        assert "Corner Market" not in audit_text
        assert "groceries" not in audit_text
        rows[2].outcome = "tampered"
        session.commit()
    assert verify(api.app.state.engine) == (False, 3)
    checked = api.client.get("/v1/audit/verify", headers=api.headers(api.admin))
    assert checked.json() == {"ok": False, "first_broken_entry_id": 3}


def test_injected_scorer_and_flag_routes(tmp_path: Path) -> None:
    class SampleScorer:
        def score(self, row: Transaction, history: list[Transaction]) -> Flag | None:
            assert row.id == "flagged"
            assert history == []
            return Flag(
                transaction_id=row.id,
                score=0.8,
                reasons=["unusual amount"],
                model_version="test-1",
                created_at=NOW,
            )

    app = create_app(
        Settings(database_url=f"sqlite:///{tmp_path / 'flags.db'}"),
        scorer=SampleScorer(),
        now=lambda: NOW,
    )
    with Session(app.state.engine) as session:
        ingest_key = create_key(session, "ingest", {"ingest"})
        read_key = create_key(session, "read", {"read"})
    with TestClient(app) as client:
        response = client.post(
            "/v1/transactions", headers={"X-API-Key": ingest_key}, json=transaction("flagged")
        )
        assert response.json()["flagged"] is True
        items = client.get("/v1/flags?account_id=acct_1", headers={"X-API-Key": read_key}).json()[
            "items"
        ]
        assert len(items) == 1
        assert items[0]["reasons"] == ["unusual amount"]
        detail = client.get(f"/v1/flags/{items[0]['id']}", headers={"X-API-Key": read_key})
        assert detail.json()["transaction_id"] == "flagged"


def test_concurrent_audit_chain(api: Fixture) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda _: api.client.get("/v1/flags", headers=api.headers(api.read)), range(20)
            )
        )
    assert all(response.status_code == 200 for response in responses)
    assert verify(api.app.state.engine) == (True, None)
    with Session(api.app.state.engine) as session:
        assert len(session.scalars(select(AuditLog)).all()) == 20


@pytest.mark.parametrize("entry", [1, 2, 3])
def test_audit_tampering_locates_entry(api: Fixture, entry: int) -> None:
    for _ in range(3):
        assert api.client.get("/v1/flags", headers=api.headers(api.read)).status_code == 200
    with Session(api.app.state.engine) as session:
        row = session.get(AuditLog, entry)
        assert row is not None
        row.status_code = 400
        session.commit()
    assert verify(api.app.state.engine) == (False, entry)


def test_verify_cli_exits_nonzero_for_broken_chain(api: Fixture, tmp_path: Path) -> None:
    assert api.client.get("/v1/flags", headers=api.headers(api.read)).status_code == 200
    with Session(api.app.state.engine) as session:
        row = session.get(AuditLog, 1)
        assert row is not None
        row.outcome = "tampered"
        session.commit()
    result = subprocess.run(
        [sys.executable, "-m", "sentinel.audit.verify"],
        capture_output=True,
        text=True,
        env=os.environ | {"SENTINEL_DATABASE_URL": f"sqlite:///{tmp_path / 'api.db'}"},
        check=False,
    )
    assert result.returncode != 0
    assert "broken at entry 1" in result.stdout
