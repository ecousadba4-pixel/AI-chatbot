"""Utilities for loading sentence-transformer embedding models."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Optional, Union

from sentence_transformers import SentenceTransformer

PathLike = Union[str, os.PathLike[str]]


def _expand_candidate_paths(
    candidate_paths: Optional[Iterable[Optional[PathLike]]],
) -> list[str]:
    """Return a filtered list of candidate paths that exist on disk."""
    if not candidate_paths:
        return []

    paths: list[str] = []
    for path in candidate_paths:
        if not path:
            continue
        expanded = os.path.expanduser(os.path.expandvars(os.fspath(path)))
        if os.path.exists(expanded):
            paths.append(expanded)
    return paths


def _default_model_search_paths(model_name: str) -> list[str]:
    """Return default directories to probe for a bundled embedding model."""

    model_relative = Path(*model_name.split("/"))
    project_root = Path(__file__).resolve().parent
    search_roots: list[Path] = []

    for env_var in ("EMBEDDING_MODEL_DIR", "APP_DATA_DIR", "DATA_DIR"):
        env_path = os.getenv(env_var)
        if env_path:
            search_roots.append(Path(env_path))

    search_roots.extend(
        [
            project_root / "Data",
            project_root / "data",
            project_root.parent / "Data",
            project_root.parent / "data",
            Path("/app/Data"),
            Path("/app/data"),
            Path("/data"),
        ]
    )

    candidates: list[str] = []
    seen: set[str] = set()
    for root in search_roots:
        candidate = root / model_relative
        candidate_str = os.fspath(candidate)
        if candidate_str not in seen:
            seen.add(candidate_str)
            candidates.append(candidate_str)
    return candidates


def _maybe_reassemble_shards(model_dir: Path) -> None:
    """Собрать файлы модели, если они были разбиты на части.

    На некоторых платформах загрузка отдельных файлов весов больше 200 МБ
    запрещена, поэтому модель может быть сохранена набором ``*.partXX`` файлов.
    ``SentenceTransformer`` ожидает итоговые файлы ``model.safetensors`` и
    ``pytorch_model.bin`` с оригинальными именами, поэтому перед загрузкой мы
    собираем их заново.
    """

    if not model_dir.is_dir():
        return

    shard_specs = {
        "model.safetensors": "model.safetensors.part*",
        "pytorch_model.bin": "pytorch_model.bin.part*",
    }

    for target_name, glob_pattern in shard_specs.items():
        target_path = model_dir / target_name
        if target_path.exists():
            continue

        parts = sorted(model_dir.glob(glob_pattern))
        if not parts:
            continue

        print(
            "🧩 Найдены части файла",
            target_name,
            f"в каталоге {model_dir} — объединяем ({len(parts)} шт.)",
        )
        temp_path = target_path.parent / f"{target_path.name}.tmp"
        with open(temp_path, "wb") as target_file:
            for part in parts:
                with open(part, "rb") as part_file:
                    shutil.copyfileobj(part_file, target_file)

        os.replace(temp_path, target_path)
        print(f"✅ Файл {target_path.name} восстановлен из частей")


def resolve_embedding_model(
    model_name: str,
    candidate_paths: Optional[Iterable[Optional[str]]] = None,
    *,
    allow_download: bool = True,
) -> SentenceTransformer:
    """Load a sentence-transformer model.

    The function first tries to load the model from the provided ``candidate_paths``.
    If none of the paths contain the model, it falls back to downloading the model by
    name using :class:`SentenceTransformer`.
    """

    search_candidates: list[Optional[PathLike]] = []
    if candidate_paths:
        search_candidates.extend(candidate_paths)
    search_candidates.extend(_default_model_search_paths(model_name))

    expanded_paths = _expand_candidate_paths(search_candidates)

    summary: list[str] = []

    if expanded_paths:
        print("🔎 Проверяем локальные пути модели:")
    for path in expanded_paths:
        print(f"  • 🔍 {path}")
        try:
            print(f"    ↪️ Пытаемся восстановить шарды модели в {path}")
            _maybe_reassemble_shards(Path(path))
        except Exception as shard_error:
            print(
                "    ⚠️ Не удалось восстановить шарды:",
                shard_error,
            )
            summary.append(f"⚠️ {path} — ошибка восстановления шардов: {shard_error}")
        else:
            summary.append(f"ℹ️ {path} — восстановление шардов не требуется или завершено")

        try:
            model = SentenceTransformer(path)
            setattr(model, "_resolved_from", path)
            print(f"🧠 Используем локальную модель эмбеддингов из {path}")
            summary.append(f"✅ {path} — модель успешно загружена")
            print("📋 Сводка по локальным путям:")
            for item in summary:
                print(f"  • {item}")
            return model
        except Exception as load_error:
            print(f"    ❌ Ошибка загрузки модели из {path}: {load_error}")
            summary.append(f"❌ {path} — ошибка загрузки: {load_error}")
            # Path exists but does not contain a valid model.
            continue

    if expanded_paths:
        print("📋 Сводка по локальным путям:")
        for item in summary:
            print(f"  • {item}")
    else:
        print("⚠️ Список локальных директорий пуст — переходим к загрузке по имени модели.")

    if not allow_download:
        searched = "\n".join(f" - {p}" for p in expanded_paths)
        raise FileNotFoundError(
            "Не удалось найти локальную модель эмбеддингов. "
            "Укажите переменную окружения EMBEDDING_MODEL_PATH (или EMBEDDING_MODEL_DIR/APP_DATA_DIR/DATA_DIR) "
            "и убедитесь, что модель располагается в одном из следующих путей:\n"
            f"{searched if searched else ' - (список путей пуст)'}"
        )

    print(
        "🌐 Локальные пути не содержат модель эмбеддингов, будет загружена по имени:",
        model_name,
    )
    model = SentenceTransformer(model_name)
    setattr(model, "_resolved_from", model_name)
    print(f"🧠 Модель эмбеддингов загружена по имени {model_name}")
    return model


