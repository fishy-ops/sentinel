from datetime import UTC, datetime

from fastapi import APIRouter, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.api.deps import EngineDep, pagination, valid_id
from sentinel.api.errors import ApiError
from sentinel.store.models import Transaction

router = APIRouter()


def transaction_dict(row: Transaction) -> dict[str, object]:
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


@router.get("/v1/accounts/{account_id}/history")
async def history(account_id: str, request: Request, db: EngineDep) -> dict[str, object]:
    valid_id(account_id, "account_id")
    request.state.resource_id = account_id
    limit, offset = pagination(request)
    before_text = request.query_params.get("before")
    before = None
    if before_text is not None:
        try:
            before = datetime.fromisoformat(before_text)
        except ValueError:
            raise ApiError(422, "validation_error", "Invalid fields: before") from None
        if before.tzinfo is None or before.utcoffset() is None:
            raise ApiError(422, "validation_error", "Invalid fields: before")
        before = before.astimezone(UTC)
    with Session(db) as session:
        query = select(Transaction).where(Transaction.account_id == account_id)
        if before is not None:
            query = query.where(Transaction.timestamp <= before)
        rows = session.scalars(
            query.order_by(Transaction.timestamp.desc(), Transaction.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return {"items": [transaction_dict(row) for row in rows], "limit": limit, "offset": offset}
