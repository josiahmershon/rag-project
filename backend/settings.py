# settings.py
from __future__ import annotations

import os
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_name: str = Field(default="postgres", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: SecretStr = Field(alias="DB_PASSWORD")

    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_model: str = Field(default="Qwen/Qwen3-32B-AWQ", alias="VLLM_MODEL")
    vllm_api_key: Optional[SecretStr] = Field(default=None, alias="VLLM_API_KEY")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

    # File upload configuration
    max_file_size: int = Field(default=10 * 1024 * 1024, alias="MAX_FILE_SIZE")
    max_query_length: int = Field(default=1000, alias="MAX_QUERY_LENGTH")
    default_chunk_size: int = Field(default=512, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=50, alias="DEFAULT_CHUNK_OVERLAP")
    default_similarity_threshold: float = Field(
        default=0.3, alias="DEFAULT_SIMILARITY_THRESHOLD"
    )

    # Observability and testing
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    test_mode: bool = Field(default=False, alias="TEST_MODE")

    @field_validator("db_password")
    @classmethod
    def _ensure_db_password(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value().strip()
        if not secret:
            raise ValueError("DB_PASSWORD must not be empty")
        return value

    @field_validator("vllm_api_key")
    @classmethod
    def _clean_vllm_api_key(cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        if value and not value.get_secret_value().strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("log_format")
    @classmethod
    def _normalize_log_format(cls, value: str) -> str:
        return value.lower()


settings = Settings()

# Ensure logging configuration picks up values even when loaded from .env
os.environ.setdefault("LOG_LEVEL", settings.log_level)
os.environ.setdefault("LOG_FORMAT", settings.log_format)