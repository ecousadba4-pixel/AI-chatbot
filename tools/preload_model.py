"""Прогрев модели эмбеддингов во время сборки контейнера."""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_on_path() -> None:
    """Добавить корень проекта в ``sys.path`` для локального запуска скрипта."""

    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_on_path()

from chatbot.cli import preload_embeddings_main


if __name__ == "__main__":  # pragma: no cover - скрипт запускается вручную
    preload_embeddings_main()
