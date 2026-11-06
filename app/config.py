from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5433/fraud"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    artifact_dir: str = "ml/artifacts/v1"
    jwt_secret: str = "dev-only-change-me"
    openrouter_api_key: str | None = None
    openrouter_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    audit_rate: float = Field(default=0.02, ge=0, le=1)
    rate_limit_per_min: int = Field(default=60, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
