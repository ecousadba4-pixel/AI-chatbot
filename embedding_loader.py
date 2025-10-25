 """Utilities for loading sentence-transformer embedding models."""
from __future__ import annotations
 
import os
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


 def resolve_embedding_model(
     model_name: str,
     candidate_paths: Optional[Iterable[Optional[str]]] = None,
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

    for path in _expand_candidate_paths(search_candidates):
         try:
             return SentenceTransformer(path)
         except Exception:
             # Path exists but does not contain a valid model.
             continue
 
     return SentenceTransformer(model_name)
