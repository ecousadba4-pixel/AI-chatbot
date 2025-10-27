"""Утилиты для загрузки модели эмбеддингов SentenceTransformer."""
from __future__ import annotations

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


def resolve_embedding_model(
    *, model_name: str, local_path: str | None = None, allow_download: bool = True
) -> SentenceTransformer:
    """Загрузить модель эмбеддингов из локального каталога или Hugging Face."""

    if local_path:
        path = Path(local_path).expanduser()
        if not path.exists():
            raise RuntimeError(
                f"Указанный путь в EMBEDDING_MODEL_LOCAL_PATH не найден: {path}"
            )

        print(f"📁 Загружаем модель эмбеддингов из локального каталога: {path}")
        try:
            model = SentenceTransformer(str(path), local_files_only=True)
        except Exception as exc:  # pragma: no cover - зависит от содержимого каталога
            raise RuntimeError(
                "Не удалось загрузить модель эмбеддингов из локального каталога. "
                "Убедитесь, что в директории находятся файлы SentenceTransformer."
            ) from exc

        setattr(model, "_resolved_from", str(path))
        print("✅ Модель эмбеддингов загружена из локального каталога")
        return model

    if not allow_download:
        raise FileNotFoundError(
            "Загрузка модели из облака запрещена, альтернативные источники не предусмотрены."
        )

    print(f"🌐 Загружаем модель эмбеддингов из Hugging Face: {model_name}")

    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    load_kwargs = {"cache_folder": cache_dir} if cache_dir else {}

    try:
        model = SentenceTransformer(model_name, **load_kwargs)
    except Exception as exc:  # pragma: no cover - сетевые ошибки
        raise RuntimeError(
            "Не удалось загрузить модель эмбеддингов из Hugging Face. "
            "Проверьте доступ к huggingface.co или предварительно скачайте веса "
            "и укажите путь через EMBEDDING_MODEL_LOCAL_PATH."
        ) from exc

    setattr(model, "_resolved_from", model_name)

    print(
        f"✅ Модель эмбеддингов из Hugging Face '{model_name}' загружена и готова к работе"
    )
    return model
