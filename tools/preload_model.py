"""Preload the embedding model during container build."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_project_on_path() -> None:
    """Добавить корень проекта в ``sys.path``."""

    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_on_path()

from embedding_loader import resolve_embedding_model


def main() -> None:
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

    # Run a dummy encode with e5-совместимыми префиксами, чтобы прогреть модель.
    model.encode(["query: warmup", "passage: warmup"])


if __name__ == "__main__":
    main()
