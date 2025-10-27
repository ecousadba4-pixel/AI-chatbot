"""Вспомогательные утилиты для работы с Amvera API."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import requests

from config import Settings


LOGGER = logging.getLogger("chatbot.amvera")


@dataclass(slots=True)
class AmveraError(Exception):
    """Специальное исключение для ошибок Amvera API."""

    message: str
    status_code: int | None = None
    details: Any | None = None

    def __str__(self) -> str:  # pragma: no cover - только для удобства логирования
        return self.message


def normalize_token(raw_token: str | None) -> str:
    token = (raw_token or "").strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer ") :].lstrip()
    return token


def ensure_token(settings: Settings) -> str:
    token = normalize_token(settings.amvera_token)
    if not token:
        raise AmveraError("Не задан токен доступа AMVERA_GPT_TOKEN")
    return token


def build_headers(settings: Settings, token: str) -> dict[str, str]:
    prefix = settings.amvera_auth_prefix.strip()
    value = f"{prefix} {token}" if prefix else token
    return {
        settings.amvera_auth_header: value,
        "Content-Type": "application/json",
    }


def build_payload(model: str | None, context: str, question: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты — ассистент загородного отеля усадьбы 'Четыре Сезона'. "
                    "Отвечай гостям кратко, дружелюбно и только на основе предоставленной информации. "
                    "Если информации нет в контексте, вежливо скажи об этом."
                ),
            },
            {
                "role": "user",
                "text": f"Контекст:\n{context}\n\nВопрос гостя: {question}",
            },
        ],
    }


def perform_request(settings: Settings, token: str, payload: dict[str, Any], *, timeout: float) -> requests.Response:
    headers = build_headers(settings, token)
    return requests.post(settings.amvera_url, headers=headers, json=payload, timeout=timeout)


def log_error(response: requests.Response) -> None:
    LOGGER.warning(
        "Запрос к Amvera завершился ошибкой: %s %s",
        response.status_code,
        response.reason,
    )
    try:
        error_json = response.json()
    except ValueError:
        error_json = {"raw": response.text}
    LOGGER.warning("Тело ошибки: %s", json.dumps(error_json, ensure_ascii=False, indent=2))
    if response.status_code == 403:
        LOGGER.warning(
            "Код 403 часто означает отсутствие доступа к выбранной модели. "
            "Проверьте права доступа в Amvera или попробуйте выбрать другую модель."
        )


def extract_answer(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message") or {}
            if isinstance(message, dict):
                answer = message.get("content") or message.get("text")
                if answer:
                    return str(answer)

    fallback = data.get("output_text") or data.get("text")
    if isinstance(fallback, str) and fallback.strip():
        return fallback

    raise AmveraError("Не удалось извлечь текст ответа из ответа модели")


def cache_key(context: str, question: str) -> str:
    return hashlib.md5(f"{question}:{context}".encode()).hexdigest()


__all__ = [
    "AmveraError",
    "normalize_token",
    "ensure_token",
    "build_headers",
    "build_payload",
    "perform_request",
    "log_error",
    "extract_answer",
    "cache_key",
]
