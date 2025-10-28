"""Инициализация инфраструктурных зависимостей."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pymorphy3
import redis
from qdrant_client import QdrantClient

from .config import Settings
from .embedding_loader import resolve_embedding_model
from .local_index import LocalIndex


LOGGER = logging.getLogger("chatbot.services")


@dataclass(slots=True)
class Dependencies:
    """Собранные сервисы для работы приложения."""

    qdrant: QdrantClient | None
    redis: redis.Redis
    morph: pymorphy3.MorphAnalyzer
    embedding_model: object
    local_index: LocalIndex | None


def create_dependencies(settings: Settings) -> Dependencies:
    """Создать и сконфигурировать все внешние сервисы."""

    morph_start = perf_counter()
    morph_analyzer = pymorphy3.MorphAnalyzer()
    morph_duration = perf_counter() - morph_start
    LOGGER.info("Создание MorphAnalyzer заняло %.2f с", morph_duration)

    local_index: LocalIndex | None = None
    embedding_start = perf_counter()
    try:
        embedding_model = resolve_embedding_model(
            model_name=settings.embedding_model,
            local_path=settings.embedding_model_path,
            allow_download=True,
        )
    except Exception as exc:  # pragma: no cover - зависит от окружения
        LOGGER.warning(
            "Не удалось загрузить модель эмбеддингов '%s': %s. Включаем локальный поиск.",
            settings.embedding_model,
            exc,
        )
        index_start = perf_counter()
        local_index = LocalIndex.from_directory(
            Path(settings.local_knowledge_base_path),
            morph_analyzer,
        )
        embedding_model = local_index
        embedding_duration = perf_counter() - index_start
        LOGGER.info(
            "Локальный индекс построен за %.2f с",
            embedding_duration,
        )
    else:
        embedding_duration = perf_counter() - embedding_start
        LOGGER.info("Загрузка модели эмбеддингов заняла %.2f с", embedding_duration)

    qdrant_client: QdrantClient | None = None
    if local_index is None:
        qdrant_start = perf_counter()
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            https=settings.qdrant_https,
        )
        qdrant_duration = perf_counter() - qdrant_start
        LOGGER.info("Инициализация клиента Qdrant заняла %.2f с", qdrant_duration)

    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
    )

    return Dependencies(
        qdrant=qdrant_client,
        redis=redis_client,
        morph=morph_analyzer,
        embedding_model=embedding_model,
        local_index=local_index,
    )


__all__ = ["Dependencies", "create_dependencies"]
