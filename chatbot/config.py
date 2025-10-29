"""Загрузка и валидация настроек чат-бота."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class SettingsError(RuntimeError):
    """Исключение уровня конфигурации."""


def _read_env(name: str, *, required: bool = True, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise SettingsError(
            f"Переменная окружения {name} должна быть установлена для запуска приложения."
        )
    return value or ""


@dataclass(slots=True, frozen=True)
class Settings:
    """Иммутабельная модель настроек."""

    embedding_model: str
    embedding_model_path: str | None

    amvera_url: str
    amvera_model: str | None
    amvera_token: str | None
    amvera_auth_header: str
    amvera_auth_prefix: str

    local_knowledge_base_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        embedding_model = _read_env("EMBEDDING_MODEL_NAME")
        embedding_model_path = _resolve_embedding_path(embedding_model)

        amvera_url = _read_env("AMVERA_GPT_URL")

        return cls(
            embedding_model=embedding_model,
            embedding_model_path=embedding_model_path,
            amvera_url=amvera_url,
            amvera_model=os.getenv("AMVERA_GPT_MODEL") or None,
            amvera_token=os.getenv("AMVERA_GPT_TOKEN") or None,
            amvera_auth_header=os.getenv("AMVERA_AUTH_HEADER", "X-Auth-Token"),
            amvera_auth_prefix=os.getenv("AMVERA_AUTH_PREFIX", "Bearer"),
            local_knowledge_base_path=os.getenv("LOCAL_KNOWLEDGE_BASE_PATH", "knowledge_base"),
        )


def _resolve_embedding_path(model_name: str) -> str | None:
    local_path = os.getenv("EMBEDDING_MODEL_LOCAL_PATH")
    if local_path:
        return str(Path(local_path).expanduser())
    cache_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if not cache_home:
        return None
    cached = Path(cache_home).expanduser() / model_name.replace("/", "_")
    return str(cached) if cached.exists() else None


__all__ = ["Settings", "SettingsError"]
