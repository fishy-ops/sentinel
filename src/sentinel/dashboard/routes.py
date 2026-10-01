from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()
ASSETS = Path(__file__).parent
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ASSETS / "index.html", headers=HEADERS)


@router.get("/assets/dashboard.css", include_in_schema=False)
def css() -> FileResponse:
    return FileResponse(ASSETS / "dashboard.css", media_type="text/css", headers=HEADERS)


@router.get("/assets/dashboard.js", include_in_schema=False)
def js() -> FileResponse:
    return FileResponse(ASSETS / "dashboard.js", media_type="text/javascript", headers=HEADERS)
