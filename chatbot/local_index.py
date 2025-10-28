"""Простой tf-idf индекс для локального поиска."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .rag import SearchResult, normalize_text


LOGGER = logging.getLogger("chatbot.local_index")


@dataclass(slots=True)
class _IndexedDocument:
    collection: str
    text: str
    normalized_tokens: list[str]


class LocalIndex:
    """Tf-idf индекс, совместимый с интерфейсом SentenceTransformer."""

    def __init__(self, documents: Sequence[_IndexedDocument]):
        if not documents:
            raise RuntimeError("Локальный индекс не может быть создан без документов.")

        self._documents = list(documents)
        self._doc_count = len(self._documents)
        self._collections = tuple(sorted({doc.collection for doc in self._documents}))

        df_counter: Counter[str] = Counter()
        for doc in self._documents:
            df_counter.update(set(doc.normalized_tokens))

        vocabulary_tokens = sorted(df_counter.keys())
        self._token_to_index = {token: idx for idx, token in enumerate(vocabulary_tokens)}
        self._idf = {
            token: math.log((1 + self._doc_count) / (1 + df_counter[token])) + 1.0
            for token in vocabulary_tokens
        }
        self._dimension = len(vocabulary_tokens)

        self._doc_vectors: list[dict[int, float]] = []
        self._doc_norms: list[float] = []
        for doc in self._documents:
            vector, norm = self._encode_tokens(doc.normalized_tokens)
            self._doc_vectors.append(vector)
            self._doc_norms.append(norm)

        LOGGER.info(
            "Локальный индекс построен: %s документов, размер словаря %s",
            self._doc_count,
            self._dimension,
        )

    @classmethod
    def from_directory(cls, directory: Path, morph) -> "LocalIndex":
        resolved = directory.expanduser().resolve()
        if not resolved.exists():
            raise RuntimeError(
                f"Каталог с локальной базой знаний не найден: {resolved}"
            )

        documents: list[_IndexedDocument] = []
        for path in sorted(resolved.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - зависит от содержимого
                LOGGER.warning("Не удалось загрузить %s: %s", path, exc)
                continue

            if not isinstance(payload, list):
                LOGGER.warning("Пропускаем %s: ожидается список документов", path)
                continue

            collection = path.stem
            for raw_item in payload:
                if not isinstance(raw_item, dict):
                    continue
                text_parts = _collect_text_parts(raw_item)
                if not text_parts:
                    continue
                combined = "\n".join(text_parts)
                normalized = normalize_text(combined, morph)
                if not normalized:
                    continue
                tokens = normalized.split()
                if not tokens:
                    continue
                documents.append(
                    _IndexedDocument(
                        collection=collection,
                        text=combined,
                        normalized_tokens=tokens,
                    )
                )

        if not documents:
            raise RuntimeError(
                "Локальная база знаний пуста или не содержит валидных документов."
            )

        return cls(documents)

    @property
    def document_count(self) -> int:
        return self._doc_count

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def collections(self) -> tuple[str, ...]:
        return self._collections

    def encode(self, text: str) -> list[float]:
        sparse_vector, _ = self._encode_text(text)
        dense = [0.0] * self._dimension
        for index, weight in sparse_vector.items():
            dense[index] = weight
        return dense

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def search(
        self, text: str, *, limit: int = 5
    ) -> tuple[list[SearchResult], list[float]]:
        sparse_query, query_norm = self._encode_text(text)
        if not sparse_query or query_norm == 0.0:
            return [], [0.0] * self._dimension

        scored: list[tuple[float, _IndexedDocument]] = []
        for document, vector, norm in zip(
            self._documents, self._doc_vectors, self._doc_norms
        ):
            if norm == 0.0:
                continue
            score = _cosine_similarity(sparse_query, vector, query_norm, norm)
            if score <= 0.0:
                continue
            scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        limited = scored[:limit]

        results = [
            SearchResult(collection=doc.collection, score=score, text=doc.text)
            for score, doc in limited
        ]
        dense_query = [0.0] * self._dimension
        for index, weight in sparse_query.items():
            dense_query[index] = weight
        return results, dense_query

    def _encode_text(self, text: str) -> tuple[dict[int, float], float]:
        tokens = text.split()
        return self._encode_tokens(tokens)

    def _encode_tokens(self, tokens: Iterable[str]) -> tuple[dict[int, float], float]:
        counts = Counter(tokens)
        total = sum(counts.values())
        if total == 0:
            return {}, 0.0

        vector: dict[int, float] = {}
        norm_sq = 0.0
        for token, count in counts.items():
            index = self._token_to_index.get(token)
            if index is None:
                continue
            tf = count / total
            idf = self._idf.get(token)
            if idf is None:
                continue
            weight = tf * idf
            if weight == 0.0:
                continue
            vector[index] = weight
            norm_sq += weight * weight

        return vector, math.sqrt(norm_sq)


def _collect_text_parts(item: dict) -> list[str]:
    parts: list[str] = []
    for key in ("title", "text", "question", "answer"):
        value = item.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                parts.append(stripped)
    keywords = item.get("keywords")
    if isinstance(keywords, (list, tuple)):
        keyword_values = [str(keyword).strip() for keyword in keywords if keyword]
        if keyword_values:
            parts.append("Ключевые слова: " + ", ".join(keyword_values))
    return parts


def _cosine_similarity(
    lhs: dict[int, float],
    rhs: dict[int, float],
    lhs_norm: float,
    rhs_norm: float,
) -> float:
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    dot = 0.0
    if len(lhs) < len(rhs):
        iterator = lhs.items()
        lookup = rhs
    else:
        iterator = rhs.items()
        lookup = lhs
    for index, weight in iterator:
        other = lookup.get(index)
        if other is None:
            continue
        dot += weight * other
    if dot <= 0.0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


__all__ = ["LocalIndex"]
