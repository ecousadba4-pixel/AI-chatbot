"""Загрузка и валидация настроек приложения."""
from __future__ import annotations

from dataclasses import dataclass
import os


def _get_env(name: str, *, required: bool = True) -> str | None:
    value = os.getenv(name)
    if required and (value is None or value == ""):
        raise RuntimeError(
            f"Переменная окружения {name} должна быть установлена для запуска приложения."
        )
    return value


def _parse_int(value: str | None, *, name: str, default: int | None = None) -> int:
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(f"Переменная окружения {name} должна быть установлена.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - защитная проверка
        raise RuntimeError(
            f"Переменная окружения {name} должна быть целым числом, получено: {value!r}"
        ) from exc


def _parse_bool(value: str | None, *, default: bool | None = None) -> bool:
    if value is None:
        if default is not None:
            return default
        raise RuntimeError(
            "Переменная окружения QDRANT_HTTPS должна быть установлена для запуска приложения."
        )
    return value.strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Иммутабельная модель настроек приложения."""

    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: str | None
    qdrant_https: bool

    redis_host: str
    redis_port: int

    embedding_model: str

    amvera_url: str
    amvera_model: str | None
    amvera_token: str | None
    amvera_auth_header: str
    amvera_auth_prefix: str

    default_collections: tuple[str, ...] = ("hotel_knowledge",)

    @classmethod
    def from_env(cls) -> "Settings":
        """Собрать настройки из переменных окружения с валидацией."""

        qdrant_host = _get_env("QDRANT_HOST", required=False) or "localhost"
        qdrant_port = _parse_int(
            os.getenv("QDRANT_PORT"), name="QDRANT_PORT", default=6333
        )
        qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
        qdrant_https = _parse_bool(
            os.getenv("QDRANT_HTTPS"), default=False
        )

        redis_host = _get_env("REDIS_HOST", required=False) or "localhost"
        redis_port = _parse_int(
            os.getenv("REDIS_PORT"), name="REDIS_PORT", default=6379
        )

        embedding_model = _get_env("EMBEDDING_MODEL_NAME")
        if embedding_model is None:
            raise RuntimeError(
                "Переменная окружения EMBEDDING_MODEL_NAME должна быть установлена для запуска приложения."
            )

        amvera_url = _get_env("AMVERA_GPT_URL")
        if amvera_url is None:
            raise RuntimeError(
                "Переменная окружения AMVERA_GPT_URL должна быть установлена для запуска приложения."
            )

        return cls(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key,
            qdrant_https=qdrant_https,
            redis_host=redis_host,
            redis_port=redis_port,
            embedding_model=embedding_model,
            amvera_url=amvera_url,
            amvera_model=os.getenv("AMVERA_GPT_MODEL") or None,
            amvera_token=os.getenv("AMVERA_GPT_TOKEN") or None,
            amvera_auth_header=os.getenv("AMVERA_AUTH_HEADER", "X-Auth-Token"),
            amvera_auth_prefix=os.getenv("AMVERA_AUTH_PREFIX", "Bearer"),
        )


__all__ = ["Settings"]
