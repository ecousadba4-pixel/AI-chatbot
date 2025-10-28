"""Загрузка и валидация настроек чат-бота."""
from __future__ import annotations

from dataclasses import dataclass, field
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


def _read_int(name: str, *, default: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        if default is None:
            raise SettingsError(
                f"Переменная окружения {name} должна быть установлена для запуска приложения."
            )
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - валидация среды
        raise SettingsError(
            f"Переменная окружения {name} должна быть целым числом, получено: {raw!r}"
        ) from exc


def _read_bool(name: str, *, default: bool | None = None) -> bool:
    raw = os.getenv(name)
    if raw is None:
        if default is None:
            raise SettingsError(
                f"Переменная окружения {name} должна быть установлена для запуска приложения."
            )
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _read_collections(value: str | None, *, name: str) -> tuple[str, ...]:
    if not value:
        raise SettingsError(
            f"Переменная окружения {name} должна содержать хотя бы одно имя коллекции."
        )
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SettingsError(
            f"Переменная окружения {name} должна содержать хотя бы одно имя коллекции."
        )
    return tuple(dict.fromkeys(items))


@dataclass(slots=True, frozen=True)
class Settings:
    """Иммутабельная модель настроек."""

    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str | None
    qdrant_https: bool

    redis_host: str
    redis_port: int

    embedding_model: str
    embedding_model_path: str | None

    amvera_url: str
    amvera_model: str | None
    amvera_token: str | None
    amvera_auth_header: str
    amvera_auth_prefix: str

    local_knowledge_base_path: str
    default_collections: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "Settings":
        qdrant_host = _read_env("QDRANT_HOST", required=False, default="localhost")
        qdrant_port = _read_int("QDRANT_PORT", default=6333)
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        qdrant_https = _read_bool("QDRANT_HTTPS", default=False)

        redis_host = _read_env("REDIS_HOST", required=False, default="localhost")
        redis_port = _read_int("REDIS_PORT", default=6379)

        embedding_model = _read_env("EMBEDDING_MODEL_NAME")
        embedding_model_path = _resolve_embedding_path(embedding_model)

        amvera_url = _read_env("AMVERA_GPT_URL")
        default_collections = _read_collections(
            os.getenv("QDRANT_COLLECTION"), name="QDRANT_COLLECTION"
        )

        return cls(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key,
            qdrant_https=qdrant_https,
            redis_host=redis_host,
            redis_port=redis_port,
            embedding_model=embedding_model,
            embedding_model_path=embedding_model_path,
            amvera_url=amvera_url,
            amvera_model=os.getenv("AMVERA_GPT_MODEL") or None,
            amvera_token=os.getenv("AMVERA_GPT_TOKEN") or None,
            amvera_auth_header=os.getenv("AMVERA_AUTH_HEADER", "X-Auth-Token"),
            amvera_auth_prefix=os.getenv("AMVERA_AUTH_PREFIX", "Bearer"),
            local_knowledge_base_path=os.getenv("LOCAL_KNOWLEDGE_BASE_PATH", "Qdrant JSON"),
            default_collections=default_collections,
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
