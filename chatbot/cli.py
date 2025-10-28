"""CLI-команды для обслуживания чат-бота."""
from __future__ import annotations

import os
from typing import Iterable

from .embedding_loader import resolve_embedding_model


def _warmup_sequences() -> Iterable[str]:
    """Возвращает последовательности для прогрева модели."""

    # Для e5-совместимых моделей важно различать query/passages.
    return ("query: warmup", "passage: warmup")


def preload_embeddings_main() -> None:
    """Загрузить модель эмбеддингов и выполнить прогрев."""

    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not model_name:
        raise RuntimeError(
            "Переменная окружения EMBEDDING_MODEL_NAME должна быть установлена перед прогревом модели."
        )

    model = resolve_embedding_model(
        model_name=model_name,
        local_path=os.getenv("EMBEDDING_MODEL_LOCAL_PATH") or None,
        allow_download=True,
    )

    sequences = list(_warmup_sequences())
    model.encode(sequences)


__all__ = ["preload_embeddings_main"]
