from datetime import UTC

from fastapi import APIRouter, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.api.deps import EngineDep, pagination, valid_id
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
    with Session(db) as session:
        rows = session.scalars(
            select(Transaction)
            .where(Transaction.account_id == account_id)
            .order_by(Transaction.timestamp.desc(), Transaction.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
        return {"items": [transaction_dict(row) for row in rows], "limit": limit, "offset": offset}
