"""Preload the embedding model during container build."""
from __future__ import annotations

import os
from pathlib import Path

from embedding_loader import resolve_embedding_model


def _default_target_directory(model_name: str) -> Path:
    model_relative = Path(*model_name.split("/"))
    root = Path(os.getenv("APP_DATA_DIR", "/app/Data"))
    return root / model_relative


def main() -> None:
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "sberbank-ai/sbert_large_nlu_ru")
    target_directory = Path(
        os.getenv("EMBEDDING_MODEL_PATH", os.fspath(_default_target_directory(model_name)))
    )

    candidate_paths = [target_directory]

    model = resolve_embedding_model(model_name=model_name, candidate_paths=candidate_paths)

    # Persist the model inside the application so the runtime can reuse the bundle.
    target_directory.parent.mkdir(parents=True, exist_ok=True)
    model.save(os.fspath(target_directory))

    # Run a dummy encode to make sure the model weights are loaded into the cache.
    model.encode(["warmup"])


if __name__ == "__main__":
    main()
