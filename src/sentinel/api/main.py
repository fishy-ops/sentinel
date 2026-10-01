import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint

from sentinel.api.auth import authenticate
from sentinel.api.rate import TokenBucket
from sentinel.api.schemas import ID, TransactionIn
from sentinel.api.settings import Settings
from sentinel.audit.chain import append, verify
from sentinel.store.db import make_engine
from sentinel.store.models import Account, ApiKey, Flag, IdempotencyKey, Transaction


class Scorer(Protocol):
    def score(self, transaction: Transaction, history: list[Transaction]) -> Flag | None: ...


class NoOpScorer:
    def score(self, transaction: Transaction, history: list[Transaction]) -> Flag | None:
        return None


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request.state.request_id}},
        status_code=status,
    )


def _scope(method: str, path: str) -> str | None:
    if method == "POST" and path in {"/v1/transactions", "/v1/transactions/batch"}:
        return "ingest"
    if method == "GET" and path == "/v1/audit/verify":
        return "admin"
    if method == "GET" and (path.startswith("/v1/accounts/") or path.startswith("/v1/flags")):
        return "read"
    return None


def _action(method: str, path: str) -> str:
    if path == "/v1/transactions/batch":
        return "transactions.batch"
    if path == "/v1/transactions":
        return "transactions.create"
    if path.startswith("/v1/accounts/"):
        return "accounts.history"
    if path == "/v1/audit/verify":
        return "audit.verify"
    if path.startswith("/v1/flags"):
        return "flags.read"
    return "unknown"


def _transaction_dict(row: Transaction) -> dict[str, object]:
    return {
        "transaction_id": row.id,
        "account_id": row.account_id,
        "timestamp": row.timestamp.replace(tzinfo=UTC).isoformat(),
        "amount": f"{row.amount:.2f}",
        "currency": row.currency,
        "merchant_name": row.merchant_name,
        "merchant_category": row.merchant_category,
        "country": row.country,
        "device_id": row.device_id,
        "channel": row.channel,
        "memo": row.memo,
    }


def _flag_dict(row: Flag) -> dict[str, object]:
    return {
        "id": row.id,
        "transaction_id": row.transaction_id,
        "score": row.score,
        "reasons": row.reasons,
        "model_version": row.model_version,
        "created_at": row.created_at.isoformat(),
    }


def create_app(
    settings: Settings | None = None,
    scorer: Scorer | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    scorer = scorer or NoOpScorer()
    limiter = TokenBucket(settings.rate_per_second, settings.rate_burst, clock)
    app = FastAPI()
    app.state.engine = engine

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def request_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(part) for part in item["loc"]) for item in exc.errors()]
        return _error(request, 422, "validation_error", f"Invalid fields: {', '.join(fields)}")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "method_not_allowed"
        message = "Not found" if exc.status_code == 404 else "Method not allowed"
        return _error(request, exc.status_code, code, message)

    @app.middleware("http")
    async def guard(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = str(uuid.uuid4())
        request.state.resource_id = None
        path = request.url.path
        if path == "/v1/healthz":
            response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        key: ApiKey | None = None
        prefix: str | None = None
        remaining: int | None = None
        try:
            with Session(engine) as session:
                key, prefix = authenticate(session, request.headers.get("X-API-Key"))
                if key is not None:
                    session.expunge(key)
            identity = (
                f"key:{key.id}"
                if key
                else f"ip:{request.client.host if request.client else 'unknown'}"
            )
            allowed, remaining, retry = limiter.consume(identity)
            if not allowed:
                response = _error(request, 429, "rate_limited", "Rate limit exceeded")
                response.headers["Retry-After"] = str(retry)
            elif key is None:
                response = _error(request, 401, "unauthorized", "Invalid or missing API key")
            elif (required := _scope(request.method, path)) and required not in key.scopes.split(
                ","
            ):
                response = _error(request, 403, "forbidden", "Insufficient scope")
            elif (length := request.headers.get("content-length")) and int(
                length
            ) > settings.max_body_bytes:
                response = _error(request, 413, "body_too_large", "Request body too large")
            elif (
                request.method in {"POST", "PUT", "PATCH"}
                and len(await request.body()) > settings.max_body_bytes
            ):
                response = _error(request, 413, "body_too_large", "Request body too large")
            else:
                request.state.api_key = key
                response = await call_next(request)
        except Exception:
            response = _error(request, 500, "internal_error", "Internal server error")
        response.headers["X-Request-ID"] = request.state.request_id
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        if path.startswith("/v1/"):
            append(
                engine,
                request.state.request_id,
                prefix if key else None,
                _action(request.method, path),
                request.state.resource_id,
                "success" if response.status_code < 400 else "failure",
                response.status_code,
            )
        return response

    def parse_transactions(body: bytes, batch: bool) -> list[TransactionIn]:
        try:
            raw = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "invalid_json", "Invalid JSON body") from None
        if batch and (not isinstance(raw, list) or not 1 <= len(raw) <= 500):
            raise ApiError(422, "validation_error", "Batch must contain 1 to 500 transactions")
        if not batch and not isinstance(raw, dict):
            raise ApiError(422, "validation_error", "Expected a transaction object")
        rows = raw if batch else [raw]
        result = []
        for index, row in enumerate(rows):
            try:
                result.append(
                    TransactionIn.model_validate_json(json.dumps(row), context={"now": now()})
                )
            except ValidationError as exc:
                fields = [
                    ".".join(str(part) for part in item["loc"]) or "timestamp"
                    for item in exc.errors()
                ]
                path = f"[{index}]." if batch else ""
                raise ApiError(
                    422, "validation_error", f"Invalid fields: {path}{', '.join(fields)}"
                ) from None
        return result

    def ingest(request: Request, body: bytes, batch: bool) -> JSONResponse:
        token = request.headers.get("Idempotency-Key")
        if token is not None and (
            not 1 <= len(token) <= 128 or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ApiError(422, "validation_error", "Invalid Idempotency-Key")
        digest = hashlib.sha256(body).hexdigest()
        key: ApiKey = request.state.api_key
        with Session(engine) as session:
            if token:
                existing = session.scalar(
                    select(IdempotencyKey).where(
                        IdempotencyKey.key_id == key.id, IdempotencyKey.token == token
                    )
                )
                if existing:
                    if existing.body_hash != digest:
                        raise ApiError(
                            422,
                            "idempotency_conflict",
                            "Idempotency-Key was used with a different body",
                        )
                    return JSONResponse(existing.response, status_code=existing.status_code)
            rows = parse_transactions(body, batch)
            if len({row.transaction_id for row in rows}) != len(rows):
                raise ApiError(409, "duplicate_transaction", "Duplicate transaction_id")
            ids = [row.transaction_id for row in rows]
            if session.scalar(
                select(func.count()).select_from(Transaction).where(Transaction.id.in_(ids))
            ):
                raise ApiError(409, "duplicate_transaction", "Duplicate transaction_id")
            output = []
            try:
                for row in rows:
                    session.execute(
                        sqlite_insert(Account)
                        .values(id=row.account_id, created_at=now())
                        .on_conflict_do_nothing(index_elements=[Account.id])
                    )
                    timestamp = row.timestamp.astimezone(UTC)
                    history = session.scalars(
                        select(Transaction)
                        .where(
                            Transaction.account_id == row.account_id,
                            Transaction.timestamp < timestamp,
                        )
                        .order_by(Transaction.timestamp.desc())
                        .limit(200)
                    ).all()
                    transaction = Transaction(
                        id=row.transaction_id,
                        account_id=row.account_id,
                        timestamp=timestamp,
                        amount=Decimal(row.amount),
                        currency=row.currency,
                        merchant_name=row.merchant_name,
                        merchant_category=row.merchant_category,
                        country=row.country,
                        device_id=row.device_id,
                        channel=row.channel,
                        memo=row.memo,
                    )
                    session.add(transaction)
                    session.flush()
                    flag = scorer.score(transaction, list(history))
                    if flag is not None:
                        flag.transaction_id = transaction.id
                        session.add(flag)
                    output.append({"transaction_id": transaction.id, "flagged": flag is not None})
                response = {"items": output} if batch else output[0]
                if token:
                    session.add(
                        IdempotencyKey(
                            key_id=key.id,
                            token=token,
                            body_hash=digest,
                            status_code=201,
                            response=response,
                        )
                    )
                session.commit()
            except IntegrityError:
                session.rollback()
                if token:
                    existing = session.scalar(
                        select(IdempotencyKey).where(
                            IdempotencyKey.key_id == key.id, IdempotencyKey.token == token
                        )
                    )
                    if existing:
                        if existing.body_hash != digest:
                            raise ApiError(
                                422,
                                "idempotency_conflict",
                                "Idempotency-Key was used with a different body",
                            ) from None
                        return JSONResponse(existing.response, status_code=existing.status_code)
                raise ApiError(409, "duplicate_transaction", "Duplicate transaction_id") from None
        request.state.resource_id = None if batch else output[0]["transaction_id"]
        return JSONResponse(response, status_code=201)

    @app.post("/v1/transactions")
    async def post_transaction(request: Request) -> JSONResponse:
        return ingest(request, await request.body(), False)

    @app.post("/v1/transactions/batch")
    async def post_batch(request: Request) -> JSONResponse:
        return ingest(request, await request.body(), True)

    def pagination(request: Request) -> tuple[int, int]:
        try:
            limit = int(request.query_params.get("limit", "50"))
            offset = int(request.query_params.get("offset", "0"))
        except ValueError:
            raise ApiError(422, "validation_error", "Invalid fields: limit, offset") from None
        if not 1 <= limit <= 200 or offset < 0:
            raise ApiError(422, "validation_error", "Invalid fields: limit, offset")
        return limit, offset

    @app.get("/v1/accounts/{account_id}/history")
    async def history(account_id: str, request: Request) -> dict[str, object]:
        if re.fullmatch(ID, account_id) is None:
            raise ApiError(422, "validation_error", "Invalid fields: account_id")
        request.state.resource_id = account_id
        limit, offset = pagination(request)
        with Session(engine) as session:
            rows = session.scalars(
                select(Transaction)
                .where(Transaction.account_id == account_id)
                .order_by(Transaction.timestamp.desc(), Transaction.id.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return {
                "items": [_transaction_dict(row) for row in rows],
                "limit": limit,
                "offset": offset,
            }

    @app.get("/v1/flags")
    async def flags(request: Request) -> dict[str, object]:
        limit, offset = pagination(request)
        account_id = request.query_params.get("account_id")
        if account_id is not None and re.fullmatch(ID, account_id) is None:
            raise ApiError(422, "validation_error", "Invalid fields: account_id")
        with Session(engine) as session:
            query = select(Flag)
            if account_id is not None:
                query = query.join(Transaction).where(Transaction.account_id == account_id)
            rows = session.scalars(
                query.order_by(Flag.created_at.desc(), Flag.id.desc()).limit(limit).offset(offset)
            ).all()
            return {"items": [_flag_dict(row) for row in rows], "limit": limit, "offset": offset}

    @app.get("/v1/flags/{flag_id}")
    async def flag_detail(flag_id: int, request: Request) -> dict[str, object]:
        with Session(engine) as session:
            row = session.get(Flag, flag_id)
            if row is None:
                raise ApiError(404, "not_found", "Flag not found")
            request.state.resource_id = str(flag_id)
            return _flag_dict(row)

    @app.get("/v1/audit/verify")
    async def audit_verify() -> dict[str, object]:
        ok, broken = verify(engine)
        return {"ok": ok, "first_broken_entry_id": broken}

    @app.get("/v1/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def app() -> FastAPI:
    return create_app()
