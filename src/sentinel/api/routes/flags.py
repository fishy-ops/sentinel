from fastapi import APIRouter, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.agent.explainer import explain_flag
from sentinel.agent.tools import BoundTools
from sentinel.api.deps import EngineDep, pagination, valid_id
from sentinel.api.errors import ApiError
from sentinel.api.routes.accounts import transaction_dict
from sentinel.store.models import Explanation, Flag, Transaction

router = APIRouter()


def flag_dict(
    row: Flag, transaction: Transaction, explanation: Explanation | None
) -> dict[str, object]:
    return {
        "id": row.id,
        "transaction_id": row.transaction_id,
        "score": row.score,
        "reasons": row.reasons,
        "model_version": row.model_version,
        "created_at": row.created_at.isoformat(),
        "transaction": {
            "id": transaction.id,
            "account": transaction.account_id,
            **transaction_dict(transaction),
        },
        "explanation": {
            "exists": explanation is not None,
            "grounded": bool(explanation and explanation.payload.get("grounded")),
        },
    }


@router.get("/v1/flags")
async def flags(request: Request, db: EngineDep) -> dict[str, object]:
    limit, offset = pagination(request)
    account_id = request.query_params.get("account_id")
    min_score_text = request.query_params.get("min_score")
    sort = request.query_params.get("sort", "transaction_time")
    if sort not in {"transaction_time", "created_at"}:
        raise ApiError(422, "validation_error", "Invalid fields: sort")
    try:
        min_score = float(min_score_text) if min_score_text is not None else 0.0
    except ValueError:
        raise ApiError(422, "validation_error", "Invalid fields: min_score") from None
    if not 0 <= min_score <= 1 or min_score != min_score:
        raise ApiError(422, "validation_error", "Invalid fields: min_score")
    if account_id is not None:
        valid_id(account_id, "account_id")
    with Session(db) as session:
        query = select(Flag, Transaction).join(Transaction).where(Flag.score >= min_score)
        if account_id is not None:
            query = query.where(Transaction.account_id == account_id)
        order = Transaction.timestamp if sort == "transaction_time" else Flag.created_at
        rows = session.execute(
            query.order_by(order.desc(), Flag.id.desc()).limit(limit).offset(offset)
        ).all()
        latest = {
            row.flag_id: row
            for row in session.scalars(
                select(Explanation)
                .where(Explanation.flag_id.in_([flag.id for flag, _ in rows]))
                .order_by(Explanation.id)
            )
        }
        return {
            "items": [
                flag_dict(flag, transaction, latest.get(flag.id)) for flag, transaction in rows
            ],
            "limit": limit,
            "offset": offset,
        }


@router.get("/v1/flags/{flag_id}")
async def flag_detail(flag_id: int, request: Request, db: EngineDep) -> dict[str, object]:
    with Session(db) as session:
        row = session.get(Flag, flag_id)
        if row is None:
            raise ApiError(404, "not_found", "Flag not found")
        request.state.resource_id = str(flag_id)
        transaction = session.get(Transaction, row.transaction_id)
        explanation = session.scalar(
            select(Explanation)
            .where(Explanation.flag_id == flag_id)
            .order_by(Explanation.id.desc())
        )
        return flag_dict(row, transaction, explanation)


@router.post("/v1/flags/{flag_id}/explain")
async def explain(flag_id: int, request: Request, db: EngineDep) -> dict[str, object]:
    with Session(db) as session:
        flag = session.get(Flag, flag_id)
        if flag is None:
            raise ApiError(404, "not_found", "Flag not found")
        transaction = session.get(Transaction, flag.transaction_id)
        bound = BoundTools(session, flag, transaction)
        payload = explain_flag(bound, request.app.state.chat_client)
        row = Explanation(
            flag_id=flag_id,
            model=payload["model"],
            payload=payload,
            created_at=request.app.state.now(),
        )
        session.add(row)
        session.commit()
        request.state.resource_id = str(flag_id)
        return {"id": row.id, "flag_id": flag_id, **payload}


@router.get("/v1/flags/{flag_id}/explanation")
async def latest_explanation(flag_id: int, request: Request, db: EngineDep) -> dict[str, object]:
    with Session(db) as session:
        if session.get(Flag, flag_id) is None:
            raise ApiError(404, "not_found", "Flag not found")
        row = session.scalar(
            select(Explanation)
            .where(Explanation.flag_id == flag_id)
            .order_by(Explanation.id.desc())
        )
        if row is None:
            raise ApiError(404, "not_found", "Explanation not found")
        request.state.resource_id = str(flag_id)
        return {"id": row.id, "flag_id": flag_id, **row.payload}
