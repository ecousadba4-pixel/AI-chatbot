"""Preload the embedding model during container build."""
from __future__ import annotations

import os

from embedding_loader import resolve_embedding_model


def main() -> None:
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if not model_name:
        raise RuntimeError(
            "Переменная окружения EMBEDDING_MODEL_NAME должна быть установлена перед прогревом модели."
        )
    model = resolve_embedding_model(model_name=model_name, allow_download=True)

    # Run a dummy encode with e5-совместимыми префиксами, чтобы прогреть модель.
    model.encode(["query: warmup", "passage: warmup"])


if __name__ == "__main__":
    main()
