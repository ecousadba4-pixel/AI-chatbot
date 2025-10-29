"""Инструменты для семантического поиска."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


_LOGGER = logging.getLogger("chatbot.rag")
_WORD_PATTERN = re.compile(r"[а-яёa-z0-9]+")
_LEMMA_CACHE_ATTR = "_chatbot_lemma_cache"
_LEMMA_CACHE_MAX_SIZE = 50_000


@dataclass(slots=True)
class SearchResult:
    """Найденный документ."""

    collection: str
    score: float
    text: str


def normalize_text(text: str, morph) -> str:
    """Привести текст к набору лемм."""

    cache = _ensure_lemma_cache(morph)
    lemmas: list[str] = []
    for word in _WORD_PATTERN.findall(text.lower()):
        lemma = cache.get(word)
        if lemma is None:
            lemma = _lemmatize_word(word, morph)
            if len(cache) >= _LEMMA_CACHE_MAX_SIZE:
                cache.clear()
            cache[word] = lemma
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


def _ensure_lemma_cache(morph) -> dict[str, str]:
    cache = getattr(morph, _LEMMA_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(morph, _LEMMA_CACHE_ATTR, cache)
    return cache


def _lemmatize_word(word: str, morph) -> str:
    try:
        parsed = morph.parse(word)
    except Exception:  # pragma: no cover - страховка от редких сбоев pymorphy
        parsed = None

    return parsed[0].normal_form if parsed else word


__all__ = [
    "SearchResult",
    "normalize_text",
    "encode",
]
