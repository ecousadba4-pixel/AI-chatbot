"""Preload the embedding model during container build."""
from __future__ import annotations

import os

from embedding_loader import resolve_embedding_model


def main() -> None:
    model_name = os.getenv(
        "EMBEDDING_MODEL_NAME", "ai-forever/sbert-base-lite-nlu-ru-v2"
    )
    model = resolve_embedding_model(model_name=model_name, allow_download=True)

    # Run a dummy encode to make sure the model weights are cached.
    model.encode(["warmup"])


if __name__ == "__main__":
    main()
