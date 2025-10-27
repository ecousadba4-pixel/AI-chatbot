"""Утилиты для загрузки модели эмбеддингов SentenceTransformer."""
from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer


def resolve_embedding_model(*, model_name: str, allow_download: bool = True) -> SentenceTransformer:
    """Загрузить модель эмбеддингов напрямую из Hugging Face."""

    if not allow_download:
        raise FileNotFoundError(
            "Загрузка модели из облака запрещена, альтернативные источники не предусмотрены."
        )

    print(f"🌐 Загружаем модель эмбеддингов из Hugging Face: {model_name}")

    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    load_kwargs = {"cache_folder": cache_dir} if cache_dir else {}

    model = SentenceTransformer(model_name, **load_kwargs)
    setattr(model, "_resolved_from", model_name)
    return model
