from fastapi import APIRouter

from sentinel.api.deps import EngineDep
from sentinel.audit.chain import inspect

router = APIRouter()


@router.get("/v1/audit/verify")
async def audit_verify(db: EngineDep) -> dict[str, object]:
    ok, broken, count, head = inspect(db)
    return {"ok": ok, "first_broken_entry_id": broken, "entry_count": count, "head_hash": head}
