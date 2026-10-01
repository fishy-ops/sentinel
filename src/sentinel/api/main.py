from fastapi import FastAPI

from sentinel.api.app import Scorer, create_app

__all__ = ["Scorer", "create_app", "app"]


def app() -> FastAPI:
    return create_app()
