"""Утилиты для загрузки модели эмбеддингов SentenceTransformer."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger("chatbot.embedding_loader")


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

        LOGGER.info("Загружаем модель эмбеддингов из локального каталога: %s", path)
        try:
            model = SentenceTransformer(str(path), local_files_only=True)
        except Exception as exc:  # pragma: no cover - зависит от содержимого каталога
            raise RuntimeError(
                "Не удалось загрузить модель эмбеддингов из локального каталога. "
                "Убедитесь, что в директории находятся файлы SentenceTransformer."
            ) from exc

        setattr(model, "_resolved_from", str(path))
        LOGGER.info("Модель эмбеддингов загружена из локального каталога")
        return model

    cache_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
    if cache_home:
        LOGGER.info(
            "Пробуем загрузить модель эмбеддингов '%s' из локального кэша: %s",
            model_name,
            cache_home,
        )
        try:
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_home,
                local_files_only=True,
            )
        except OSError as exc:  # pragma: no cover - когда модели нет в кэше
            LOGGER.warning(
                "Не нашли модель '%s' в локальном кэше %s: %s",
                model_name,
                cache_home,
                exc,
            )
        except Exception as exc:  # pragma: no cover - повреждённый кэш
            LOGGER.warning(
                "Не удалось загрузить модель '%s' из локального кэша %s: %s",
                model_name,
                cache_home,
                exc,
            )
        else:
            setattr(model, "_resolved_from", f"cache://{model_name}")
            LOGGER.info("Модель эмбеддингов загружена из локального кэша")
            return model

    if not allow_download:
        raise FileNotFoundError(
            "Загрузка модели из облака запрещена, альтернативные источники не предусмотрены."
        )

    LOGGER.info("Загружаем модель эмбеддингов из Hugging Face: %s", model_name)

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

    LOGGER.info(
        "Модель эмбеддингов из Hugging Face '%s' загружена и готова к работе",
        model_name,
    )
    return model
