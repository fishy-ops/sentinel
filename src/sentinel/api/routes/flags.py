from fastapi import APIRouter, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.api.deps import EngineDep, pagination, valid_id
from sentinel.api.errors import ApiError
from sentinel.store.models import Flag, Transaction

router = APIRouter()


def flag_dict(row: Flag) -> dict[str, object]:
    return {
        "id": row.id,
        "transaction_id": row.transaction_id,
        "score": row.score,
        "reasons": row.reasons,
        "model_version": row.model_version,
        "created_at": row.created_at.isoformat(),
    }


@router.get("/v1/flags")
async def flags(request: Request, db: EngineDep) -> dict[str, object]:
    limit, offset = pagination(request)
    account_id = request.query_params.get("account_id")
    if account_id is not None:
        valid_id(account_id, "account_id")
    with Session(db) as session:
        query = select(Flag)
        if account_id is not None:
            query = query.join(Transaction).where(Transaction.account_id == account_id)
        rows = session.scalars(
            query.order_by(Flag.created_at.desc(), Flag.id.desc()).limit(limit).offset(offset)
        ).all()
        return {"items": [flag_dict(row) for row in rows], "limit": limit, "offset": offset}


@router.get("/v1/flags/{flag_id}")
async def flag_detail(flag_id: int, request: Request, db: EngineDep) -> dict[str, object]:
    with Session(db) as session:
        row = session.get(Flag, flag_id)
        if row is None:
            raise ApiError(404, "not_found", "Flag not found")
        request.state.resource_id = str(flag_id)
        return flag_dict(row)
