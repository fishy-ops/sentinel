import re
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine

from sentinel.api.errors import ApiError
from sentinel.api.schemas import ID


def engine(request: Request) -> Engine:
    return request.app.state.engine


EngineDep = Annotated[Engine, Depends(engine)]


def pagination(request: Request) -> tuple[int, int]:
    try:
        limit = int(request.query_params.get("limit", "50"))
        offset = int(request.query_params.get("offset", "0"))
    except ValueError:
        raise ApiError(422, "validation_error", "Invalid fields: limit, offset") from None
    if not 1 <= limit <= 200 or offset < 0:
        raise ApiError(422, "validation_error", "Invalid fields: limit, offset")
    return limit, offset


def valid_id(value: str, field: str) -> None:
    if re.fullmatch(ID, value) is None:
        raise ApiError(422, "validation_error", f"Invalid fields: {field}")
