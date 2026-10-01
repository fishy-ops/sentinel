import re
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import Response
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from starlette.middleware.base import RequestResponseEndpoint

from sentinel.api.auth import authenticate
from sentinel.api.errors import error
from sentinel.api.rate import TokenBucket
from sentinel.api.settings import Settings
from sentinel.audit.chain import append
from sentinel.store.models import ApiKey


def _scope(method: str, path: str) -> str | None:
    if method == "POST" and path in {"/v1/transactions", "/v1/transactions/batch"}:
        return "ingest"
    if method == "GET" and path == "/v1/audit/verify":
        return "admin"
    if method == "GET" and (path.startswith("/v1/accounts/") or path.startswith("/v1/flags")):
        return "read"
    if method == "POST" and path.startswith("/v1/flags/") and path.endswith("/explain"):
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
        if method == "POST" and path.endswith("/explain"):
            return "flags.explain"
        if path.endswith("/explanation"):
            return "flags.explanation.read"
        return "flags.read"
    return "unknown"


async def _bounded_body(request: Request, limit: int) -> bool:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return False
        chunks.append(chunk)
    request._body = b"".join(chunks)
    return True


def _too_large(length: str, limit: int) -> bool:
    normalized = length.lstrip("0") or "0"
    maximum = str(limit)
    return len(normalized) > len(maximum) or (
        len(normalized) == len(maximum) and normalized > maximum
    )


def install_middleware(app: FastAPI, settings: Settings, db: Engine, limiter: TokenBucket) -> None:
    @app.middleware("http")
    async def guard(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = str(uuid.uuid4())
        request.state.resource_id = None
        path = request.url.path
        if (
            path == "/v1/healthz"
            or path == "/"
            or path in {"/assets/dashboard.css", "/assets/dashboard.js"}
        ):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        key: ApiKey | None = None
        prefix: str | None = None
        remaining: int | None = None
        try:
            with Session(db) as session:
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
                response = error(request, 429, "rate_limited", "Rate limit exceeded")
                response.headers["Retry-After"] = str(retry)
            elif key is None:
                response = error(request, 401, "unauthorized", "Invalid or missing API key")
            elif (required := _scope(request.method, path)) and required not in key.scopes.split(
                ","
            ):
                response = error(request, 403, "forbidden", "Insufficient scope")
            else:
                length = request.headers.get("content-length")
                if length is not None and re.fullmatch(r"[0-9]+", length) is None:
                    response = error(request, 400, "bad_request", "Invalid Content-Length")
                elif length is not None and _too_large(length, settings.max_body_bytes):
                    response = error(request, 413, "body_too_large", "Request body too large")
                elif request.method in {"POST", "PUT", "PATCH"} and not await _bounded_body(
                    request, settings.max_body_bytes
                ):
                    response = error(request, 413, "body_too_large", "Request body too large")
                else:
                    request.state.api_key = key
                    response = await call_next(request)
        except Exception:
            response = error(request, 500, "internal_error", "Internal server error")
        response.headers["X-Request-ID"] = request.state.request_id
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        if path.startswith("/v1/"):
            append(
                db,
                request.state.request_id,
                prefix if prefix and re.fullmatch(r"[0-9a-fA-F]{8}", prefix) else None,
                _action(request.method, path),
                request.state.resource_id,
                "success" if response.status_code < 400 else "failure",
                response.status_code,
            )
        return response
