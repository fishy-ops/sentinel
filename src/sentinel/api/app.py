import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from fastapi import FastAPI

from sentinel.agent.explainer import ChatClient, HttpChatClient
from sentinel.api.errors import install_handlers
from sentinel.api.middleware import install_middleware
from sentinel.api.rate import TokenBucket
from sentinel.api.routes import accounts, audit, flags, health, transactions
from sentinel.api.settings import Settings
from sentinel.dashboard import routes as dashboard
from sentinel.store.db import make_engine
from sentinel.store.models import Flag, Transaction


class Scorer(Protocol):
    def score(self, transaction: Transaction, history: list[Transaction]) -> Flag | None: ...


def create_app(
    settings: Settings | None = None,
    scorer: Scorer | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    chat_client: ChatClient | None = None,
) -> FastAPI:
    settings = settings or Settings()
    db = make_engine(settings.database_url)
    if scorer is None:
        from sentinel.detect.score import CombinedScorer

        scorer = CombinedScorer.from_artifact(settings.model_artifact)
    app = FastAPI()
    app.state.engine = db
    app.state.scorer = scorer
    app.state.now = now
    app.state.chat_client = chat_client or HttpChatClient()
    install_handlers(app)
    install_middleware(
        app, settings, db, TokenBucket(settings.rate_per_second, settings.rate_burst, clock)
    )
    for router in (transactions.router, accounts.router, flags.router, audit.router, health.router):
        app.include_router(router)
    app.include_router(dashboard.router)
    return app
