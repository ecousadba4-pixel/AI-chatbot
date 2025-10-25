"""Utilities for loading sentence-transformer embedding models."""
from __future__ import annotations

import os
from typing import Iterable, Optional

from sentence_transformers import SentenceTransformer


def _expand_candidate_paths(candidate_paths: Optional[Iterable[Optional[str]]]) -> list[str]:
    """Return a filtered list of candidate paths that exist on disk."""
    if not candidate_paths:
        return []

    paths: list[str] = []
    for path in candidate_paths:
        if not path:
            continue
        expanded = os.path.expanduser(os.path.expandvars(path))
        if os.path.exists(expanded):
            paths.append(expanded)
    return paths


def resolve_embedding_model(
    model_name: str,
    candidate_paths: Optional[Iterable[Optional[str]]] = None,
) -> SentenceTransformer:
    """Load a sentence-transformer model.

    The function first tries to load the model from the provided ``candidate_paths``.
    If none of the paths contain the model, it falls back to downloading the model by
    name using :class:`SentenceTransformer`.
    """

    for path in _expand_candidate_paths(candidate_paths):
        try:
            return SentenceTransformer(path)
        except Exception:
            # Path exists but does not contain a valid model.
            continue

    return SentenceTransformer(model_name)
