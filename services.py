"""Инфраструктурные зависимости приложения."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pymorphy3
import redis
from qdrant_client import QdrantClient

from embedding_loader import resolve_embedding_model
from config import Settings


@dataclass(slots=True)
class Dependencies:
    """Собранные сервисы для работы приложения."""

    qdrant: QdrantClient
    redis: redis.Redis
    morph: pymorphy3.MorphAnalyzer
    embedding_model: object


def create_dependencies(settings: Settings) -> Dependencies:
    """Создать и сконфигурировать все внешние сервисы."""

    embedding_start = perf_counter()
    embedding_model = resolve_embedding_model(
        model_name=settings.embedding_model,
        local_path=settings.embedding_model_path,
        allow_download=True,
    )
    embedding_duration = perf_counter() - embedding_start
    print(f"⏱️ Загрузка модели эмбеддингов заняла {embedding_duration:.2f} с")

    qdrant_start = perf_counter()
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key or None,
        https=settings.qdrant_https,
    )
    qdrant_duration = perf_counter() - qdrant_start
    print(f"⏱️ Инициализация клиента Qdrant заняла {qdrant_duration:.2f} с")

    morph_start = perf_counter()
    morph_analyzer = pymorphy3.MorphAnalyzer()
    morph_duration = perf_counter() - morph_start
    print(f"⏱️ Создание MorphAnalyzer заняло {morph_duration:.2f} с")

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
    )


__all__ = ["Dependencies", "create_dependencies"]
