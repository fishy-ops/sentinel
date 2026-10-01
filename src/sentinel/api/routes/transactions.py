import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sentinel.api.deps import EngineDep
from sentinel.api.errors import ApiError
from sentinel.api.schemas import TransactionIn
from sentinel.store.models import Account, ApiKey, IdempotencyKey, Transaction

router = APIRouter()


def parse_transactions(body: bytes, batch: bool, now: datetime) -> list[TransactionIn]:
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
            result.append(TransactionIn.model_validate_json(json.dumps(row), context={"now": now}))
        except ValidationError as exc:
            fields = [
                ".".join(str(part) for part in item["loc"]) or "timestamp" for item in exc.errors()
            ]
            path = f"[{index}]." if batch else ""
            raise ApiError(
                422, "validation_error", f"Invalid fields: {path}{', '.join(fields)}"
            ) from None
    return result


def ingest(request: Request, body: bytes, batch: bool, db: Engine) -> JSONResponse:
    token = request.headers.get("Idempotency-Key")
    if token is not None and (
        not 1 <= len(token) <= 128 or any(ord(c) < 33 or ord(c) > 126 for c in token)
    ):
        raise ApiError(422, "validation_error", "Invalid Idempotency-Key")
    digest = hashlib.sha256(body).hexdigest()
    key: ApiKey = request.state.api_key
    now = request.app.state.now
    scorer = request.app.state.scorer
    with Session(db) as session:
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
        rows = parse_transactions(body, batch, now())
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
                        Transaction.account_id == row.account_id, Transaction.timestamp < timestamp
                    )
                    .order_by(Transaction.timestamp.desc())
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


@router.post("/v1/transactions")
async def post_transaction(request: Request, db: EngineDep) -> JSONResponse:
    return ingest(request, await request.body(), False, db)


@router.post("/v1/transactions/batch")
async def post_batch(request: Request, db: EngineDep) -> JSONResponse:
    return ingest(request, await request.body(), True, db)
