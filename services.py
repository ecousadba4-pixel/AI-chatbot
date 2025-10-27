"""Инфраструктурные зависимости приложения."""
from __future__ import annotations

from dataclasses import dataclass

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

    embedding_model = resolve_embedding_model(
        model_name=settings.embedding_model,
        allow_download=True,
    )

    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key or None,
        https=settings.qdrant_https,
    )

    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True,
    )

    return Dependencies(
        qdrant=qdrant_client,
        redis=redis_client,
        morph=pymorphy3.MorphAnalyzer(),
        embedding_model=embedding_model,
    )


__all__ = ["Dependencies", "create_dependencies"]
