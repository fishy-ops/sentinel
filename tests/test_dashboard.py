from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from sentinel.api.app import create_app
from sentinel.api.auth import create_key
from sentinel.api.settings import Settings
from sentinel.store.models import Account, Explanation, Flag, Transaction

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def test_dashboard_assets_are_public_and_hardened(tmp_path: Path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'dashboard.db'}"))
    with TestClient(app) as client:
        for path, content_type in (
            ("/", "text/html"),
            ("/assets/dashboard.css", "text/css"),
            ("/assets/dashboard.js", "text/javascript"),
        ):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(content_type)
            assert response.headers["content-security-policy"] == (
                "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            )
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["referrer-policy"] == "no-referrer"
        assert "sk_" not in client.get("/").text
        script = client.get("/assets/dashboard.js").text
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert sink not in script


def test_flag_transaction_status_and_min_score(tmp_path: Path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'flags.db'}"))
    markup = "<img src=x onerror=alert(1)>"
    with Session(app.state.engine) as session:
        session.add(Account(id="acct_1", created_at=NOW))
        for index, score in enumerate((0.4, 0.9), 1):
            transaction = Transaction(
                id=f"tx_{index}",
                account_id="acct_1",
                timestamp=NOW,
                amount=Decimal("25.25"),
                currency="USD",
                merchant_name=markup,
                merchant_category="retail",
                country="US",
                device_id="dev_1",
                channel="online",
                memo=markup,
            )
            session.add(transaction)
            session.flush()
            flag = Flag(
                transaction_id=transaction.id,
                score=score,
                reasons=["review"],
                model_version="test",
                created_at=NOW,
            )
            session.add(flag)
            session.flush()
            if index == 2:
                session.add(
                    Explanation(
                        flag_id=flag.id,
                        model="test",
                        created_at=NOW,
                        payload={
                            "grounded": True,
                            "explanation": None,
                            "cited_records": {"tx_2": {"ref": "tx_2"}},
                        },
                    )
                )
        key = create_key(session, "read", {"read"})
    with TestClient(app) as client:
        headers = {"X-API-Key": key}
        response = client.get("/v1/flags?min_score=0.7", headers=headers)
        assert response.status_code == 200
        item = response.json()["items"][0]
        assert len(response.json()["items"]) == 1
        assert item["transaction"]["id"] == "tx_2"
        assert item["transaction"]["account"] == "acct_1"
        assert item["transaction"]["merchant_name"] == markup
        assert item["transaction"]["memo"] == markup
        assert item["explanation"] == {"exists": True, "grounded": True}
        all_items = client.get("/v1/flags?account_id=acct_1", headers=headers).json()["items"]
        assert len(all_items) == 2
        assert all_items[1]["explanation"] == {"exists": False, "grounded": False}
        assert client.get(f"/v1/flags/{item['id']}", headers=headers).json() == item
        assert client.get(f"/v1/flags/{item['id']}/explanation", headers=headers).json()[
            "cited_records"
        ]["tx_2"] == {"ref": "tx_2"}
        for invalid in ("nan", "inf", "-0.1", "1.1", "junk"):
            assert client.get(f"/v1/flags?min_score={invalid}", headers=headers).status_code == 422


def test_flag_sort_defaults_to_transaction_time(tmp_path: Path) -> None:
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'sort.db'}"))
    with Session(app.state.engine) as session:
        session.add(Account(id="acct_1", created_at=NOW - timedelta(days=3)))
        session.flush()
        for index, age in enumerate((1, 2), 1):
            session.add(
                Transaction(
                    id=f"tx_{index}",
                    account_id="acct_1",
                    timestamp=NOW - timedelta(days=age),
                    amount=Decimal("25.25"),
                    currency="USD",
                    merchant_name="Corner Market",
                    merchant_category="grocery",
                    country="US",
                    device_id="dev_1",
                    channel="online",
                )
            )
            session.flush()
            session.add(
                Flag(
                    transaction_id=f"tx_{index}",
                    score=0.8,
                    reasons=["review"],
                    model_version="test",
                    created_at=NOW + timedelta(minutes=index),
                )
            )
        key = create_key(session, "read", {"read"})
    with TestClient(app) as client:
        headers = {"X-API-Key": key}
        for query, expected in (("", ["tx_1", "tx_2"]), ("?sort=created_at", ["tx_2", "tx_1"])):
            response = client.get(f"/v1/flags{query}", headers=headers)
            assert response.status_code == 200
            assert [row["transaction_id"] for row in response.json()["items"]] == expected
        assert client.get("/v1/flags?sort=wrong", headers=headers).status_code == 422
