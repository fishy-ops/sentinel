from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message


def error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request.state.request_id}},
        status_code=status,
    )


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def request_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(part) for part in item["loc"]) for item in exc.errors()]
        return error(request, 422, "validation_error", f"Invalid fields: {', '.join(fields)}")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        known = {
            400: ("bad_request", "Bad request"),
            401: ("unauthorized", "Unauthorized"),
            403: ("forbidden", "Forbidden"),
            404: ("not_found", "Not found"),
            405: ("method_not_allowed", "Method not allowed"),
            413: ("body_too_large", "Request body too large"),
            422: ("validation_error", "Validation error"),
            429: ("rate_limited", "Rate limit exceeded"),
        }
        code, message = known.get(exc.status_code, ("http_error", "HTTP error"))
        return error(request, exc.status_code, code, message)
