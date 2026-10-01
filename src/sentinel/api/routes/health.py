from fastapi import APIRouter

router = APIRouter()


@router.get("/v1/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}
