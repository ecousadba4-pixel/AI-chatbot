"""Функции для поиска релевантной информации."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient


@dataclass(slots=True)
class SearchResult:
    collection: str
    score: float
    text: str


def normalize_text(text: str, morph) -> str:
    """Базовая лемматизация/нормализация вопроса."""

    import re

    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)


def encode(text: str, model) -> list[float]:
    """Кодирование текста в эмбеддинг."""

    vector = model.encode(text)
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    if isinstance(vector, Sequence):
        return list(vector)
    raise TypeError("Модель эмбеддингов вернула неподдерживаемый тип вектора")


def extract_payload_text(payload: dict[str, Any]) -> str:
    """Извлечь человекочитаемый текст из результата Qdrant."""

    text = payload.get("text") or payload.get("text_bm25") or ""
    if text:
        return text

    raw = payload.get("raw")
    if isinstance(raw, dict):
        text_blocks = raw.get("text_blocks")
        if isinstance(text_blocks, dict):
            combined = "\n".join(str(v) for v in text_blocks.values() if v)
            if combined:
                return combined

        raw_text = raw.get("text")
        if raw_text:
            return raw_text

        if raw.get("category") == "faq":
            question = raw.get("question") or ""
            answer = raw.get("answer") or ""
            if question or answer:
                return f"Вопрос: {question}\nОтвет: {answer}"

    return ""


def search_all_collections(
    client: QdrantClient,
    collections: Iterable[str],
    query_embedding: Iterable[float],
    *,
    limit: int = 5,
) -> list[SearchResult]:
    """Поиск по всем коллекциям с объединением результатов."""

    results: list[SearchResult] = []
    embedding_vector = list(query_embedding)

    for collection in collections:
        try:
            search_response = client.search(
                collection_name=collection,
                query_vector=embedding_vector,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - сетевые ошибки
            print(f"⚠️ Ошибка поиска в {collection}: {exc}")
            continue

        for hit in search_response:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    collection=collection,
                    score=hit.score,
                    text=extract_payload_text(payload),
                )
            )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


__all__ = [
    "SearchResult",
    "normalize_text",
    "encode",
    "extract_payload_text",
    "search_all_collections",
]
