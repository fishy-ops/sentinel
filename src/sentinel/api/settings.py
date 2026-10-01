from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", extra="forbid")

    database_url: str = "sqlite:///./sentinel.db"
    rate_per_second: float = Field(default=10, gt=0)
    rate_burst: int = Field(default=20, gt=0)
    max_body_bytes: int = Field(default=1_000_000, gt=0)
    model_artifact: str | None = None
