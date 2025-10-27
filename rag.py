"""Инструменты поиска релевантных ответов в Qdrant."""
from __future__ import annotations

from dataclasses import dataclass
from heapq import nlargest
import re
from typing import Any, Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient


_WORD_PATTERN = re.compile(r"[а-яёa-z0-9]+")


@dataclass(slots=True)
class SearchResult:
    """Найденный документ в Qdrant."""

    collection: str
    score: float
    text: str


def normalize_text(text: str, morph) -> str:
    """Привести текст к леммам, устойчиво к сбоям морфологического анализа."""

    lemmas: list[str] = []
    for word in _WORD_PATTERN.findall(text.lower()):
        try:
            parsed = morph.parse(word)
        except Exception:  # pragma: no cover - страховка от редких сбоев pymorphy
            parsed = None

        lemma = parsed[0].normal_form if parsed else word
        lemmas.append(lemma)

    return " ".join(lemmas)


def encode(text: str, model) -> list[float]:
    """Кодирование текста запроса с добавлением e5-префикса."""

    cleaned = text.strip()
    prepared = f"query: {cleaned}" if cleaned else "query:"
    vector = model.encode(prepared)
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    if isinstance(vector, Sequence):
        return list(vector)
    raise TypeError("Модель эмбеддингов вернула неподдерживаемый тип вектора")


def extract_payload_text(payload: dict[str, Any]) -> str:
    """Извлечь человекочитаемый текст из полезной нагрузки Qdrant."""

    text = payload.get("text") or payload.get("text_bm25")
    if text:
        return text

    raw = payload.get("raw")
    if isinstance(raw, dict):
        text_blocks = raw.get("text_blocks")
        if isinstance(text_blocks, dict):
            combined = "\n".join(str(value) for value in text_blocks.values() if value)
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
    """Поиск по нескольким коллекциям с агрегацией лучших результатов."""

    embedding_vector = list(query_embedding)
    aggregated: list[SearchResult] = []

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
            text = extract_payload_text(payload)
            if not text:
                continue
            aggregated.append(
                SearchResult(
                    collection=collection,
                    score=hit.score,
                    text=text,
                )
            )

    return nlargest(limit, aggregated, key=lambda item: item.score)


__all__ = [
    "SearchResult",
    "normalize_text",
    "encode",
    "extract_payload_text",
    "search_all_collections",
]
