"""Утилиты для загрузки модели эмбеддингов SentenceTransformer."""
from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer


def _should_use_local_only(*, allow_download: bool) -> tuple[bool, bool]:
    """Определить, нужно ли работать исключительно с локальным кэшем моделей."""

    offline_flags = {
        os.getenv("HF_HUB_OFFLINE"),
        os.getenv("TRANSFORMERS_OFFLINE"),
        os.getenv("SENTENCE_TRANSFORMERS_OFFLINE"),
    }

    offline_forced = any(
        flag and flag.strip().lower() not in {"0", "false"} for flag in offline_flags
    )

    if offline_forced:
        return True, True

    if not allow_download:
        return True, False

    return False, False


def resolve_embedding_model(*, model_name: str, allow_download: bool = True) -> SentenceTransformer:
    """Загрузить модель эмбеддингов из облака Hugging Face."""

    if not allow_download:
        raise FileNotFoundError(
            "Загрузка модели из облака запрещена, альтернативные источники не предусмотрены."
        )

    print(f"🌐 Загружаем модель эмбеддингов из Hugging Face: {model_name}")
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    local_only, offline_forced = _should_use_local_only(allow_download=allow_download)

    load_kwargs = {"cache_folder": cache_dir} if cache_dir else {}
    if local_only:
        load_kwargs["local_files_only"] = True

    try:
        model = SentenceTransformer(model_name, **load_kwargs)
    except (OSError, ValueError) as exc:
        if local_only and allow_download and not offline_forced:
            print(
                "⚠️ Не удалось загрузить модель из локального кэша. "
                "Пробуем скачать из Hugging Face..."
            )
            load_kwargs.pop("local_files_only", None)
            model = SentenceTransformer(model_name, **load_kwargs)
        else:
            raise RuntimeError(
                "Не удалось загрузить модель эмбеддингов. "
                "Убедитесь, что файлы присутствуют в кэше или разрешена загрузка."
            ) from exc

    setattr(model, "_resolved_from", model_name)
    return model
