"""Preload the embedding model during container build."""
from __future__ import annotations

import os

from embedding_loader import resolve_embedding_model


def main() -> None:
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "sberbank-ai/sbert_large_nlu_ru")
    candidate_paths = [
        os.getenv("EMBEDDING_MODEL_PATH"),
        "/app/data/sberbank-ai/sbert_large_nlu_ru",
        "/data/sberbank-ai/sbert_large_nlu_ru",
    ]

    model = resolve_embedding_model(model_name=model_name, candidate_paths=candidate_paths)

    # Run a dummy encode to make sure the model weights are loaded into the cache.
    model.encode(["warmup"])


if __name__ == "__main__":
    main()
