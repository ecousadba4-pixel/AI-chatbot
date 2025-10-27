import json
import os
import re
import hashlib
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pymorphy3
import redis
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from qdrant_client import QdrantClient

from embedding_loader import resolve_embedding_model
from price_dialog import handle_price_dialog

# ----------------------------
# INIT
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# ENV VARIABLES
# ----------------------------
DEFAULT_COLLECTIONS = ["hotel_knowledge"]
COLLECTIONS = DEFAULT_COLLECTIONS.copy()
ERROR_MESSAGE = "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

_qdrant_https_raw = os.getenv("QDRANT_HTTPS")
if _qdrant_https_raw is None:
    raise RuntimeError(
        "Переменная окружения QDRANT_HTTPS должна быть установлена для запуска приложения."
    )
QDRANT_HTTPS = _qdrant_https_raw.lower() in ("1", "true", "yes")

AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL")
if not AMVERA_GPT_URL:
    raise RuntimeError(
        "Переменная окружения AMVERA_GPT_URL должна быть установлена для запуска приложения."
    )
AMVERA_GPT_MODEL = os.getenv("AMVERA_GPT_MODEL")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN")
AMVERA_AUTH_HEADER = "X-Auth-Token"
AMVERA_AUTH_PREFIX = "Bearer"

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))

def _filter_existing_collections(
    client: QdrantClient,
    requested: list[str],
    fallback: list[str],
) -> list[str]:
    """Убираем из списка отсутствующие коллекции и подбираем резервные."""

    try:
        response = client.get_collections()
        available = {collection.name for collection in response.collections}
    except Exception as exc:  # pragma: no cover - сетевой сбой
        print(f"⚠️ Не удалось получить список коллекций из Qdrant: {exc}")
        return requested

    if not available:
        print("⚠️ В Qdrant не найдено ни одной коллекции")
        return requested

    filtered = [name for name in requested if name in available]
    if filtered:
        missing = [name for name in requested if name not in available]
        if missing:
            print(
                "⚠️ Пропускаем отсутствующие коллекции: " + ", ".join(sorted(missing))
            )
        return filtered

    fallback_candidates = [name for name in fallback if name in available]
    if not fallback_candidates:
        fallback_candidates = sorted(available)

    print(
        "⚠️ Ни одна из запрошенных коллекций не найдена. Используем fallback: "
        + ", ".join(fallback_candidates)
    )
    return fallback_candidates

# ----------------------------
# CONNECTIONS
# ----------------------------
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

morph = pymorphy3.MorphAnalyzer()

# >>> ГЛАВНОЕ ИЗМЕНЕНИЕ: русская модель эмбеддингов <<<
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
if not EMBEDDING_MODEL_NAME:
    raise RuntimeError(
        "Переменная окружения EMBEDDING_MODEL_NAME должна быть установлена для запуска приложения."
    )


model = resolve_embedding_model(
    model_name=EMBEDDING_MODEL_NAME,
    allow_download=True,
)

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    https=QDRANT_HTTPS,
)

print(f"✅ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} (https={QDRANT_HTTPS})")
print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
print("🔢 Embedding dimension:", model.get_sentence_embedding_dimension())
print(f"🤖 Amvera GPT endpoint: {AMVERA_GPT_URL} (model={AMVERA_GPT_MODEL})")

COLLECTIONS = _filter_existing_collections(
    qdrant_client,
    COLLECTIONS,
    DEFAULT_COLLECTIONS,
)
print("📚 Активные коллекции:", ", ".join(COLLECTIONS))

# ----------------------------
# HELPERS
# ----------------------------
def normalize_text(text: str) -> str:
    """Базовая лемматизация/нормализация вопроса (RU + латиница, цифры)."""
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)

def encode(text: str) -> list:
    """Кодирование текста в эмбеддинг (list[float]) одной и той же моделью для запросов и документов."""
    # Важно: документы в Qdrant должны быть эмбеддены этой же моделью
    vec = model.encode(text)
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    return vec

def _extract_payload_text(payload: dict[str, Any]) -> str:
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


def search_all_collections(query_embedding: Iterable[float], limit: int = 5) -> list[dict[str, Any]]:
    """Поиск по всем коллекциям с объединением результатов."""

    aggregated: list[dict[str, Any]] = []
    embedding_vector = list(query_embedding)

    for coll in COLLECTIONS:
        try:
            search = qdrant_client.search(
                collection_name=coll,
                query_vector=embedding_vector,
                limit=limit,
            )
        except Exception as exc:
            print(f"⚠️ Ошибка поиска в {coll}: {exc}")
            continue

        for hit in search:
            payload = hit.payload or {}
            aggregated.append(
                {
                    "collection": coll,
                    "score": hit.score,
                    "text": _extract_payload_text(payload),
                }
            )

    aggregated.sort(key=lambda item: item["score"], reverse=True)
    return aggregated[:limit]

def _normalize_amvera_token(raw_token: str | None) -> str:
    """Очистить токен: убрать префикс ``Bearer`` и лишние пробелы."""

    token = (raw_token or "").strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer ") :].lstrip()
    return token


def _ensure_amvera_token() -> str | None:
    """Получить токен авторизации, аналогично ensure_api_key из скрипта проверки."""

    token = _normalize_amvera_token(AMVERA_GPT_TOKEN)

    if not token:
        print("⚠️ Не задан токен доступа (AMVERA_GPT_TOKEN)")
        return None

    return token


def _build_amvera_headers(token: str) -> dict[str, str]:
    """Сформировать заголовки авторизации по соглашениям Amvera."""

    prefix = AMVERA_AUTH_PREFIX.strip()
    if prefix:
        header_value = f"{prefix} {token}"
    else:
        header_value = token

    return {
        AMVERA_AUTH_HEADER: header_value,
        "Content-Type": "application/json",
    }


def _build_amvera_payload(model: str, context: str, question: str) -> dict[str, object]:
    """Сформировать тело запроса по образцу из утилиты проверки API."""

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


def _perform_amvera_request(
    url: str,
    token: str,
    payload: dict[str, object],
    timeout: float,
) -> requests.Response:
    """Выполнить HTTP-запрос к Amvera API так же, как в тестовой утилите."""

    headers = _build_amvera_headers(token)
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


def _log_amvera_error(response: requests.Response) -> None:
    """Логирование ошибок API с подробностями по образцу скрипта проверки."""

    print(
        f"Запрос завершился ошибкой: {response.status_code} {response.reason}",
    )
    try:
        error_json = response.json()
    except ValueError:
        error_json = {"raw": response.text}

    print(json.dumps(error_json, ensure_ascii=False, indent=2))

    if response.status_code == 403:
        print(
            "Подсказка: код 403 часто означает отсутствие доступа к выбранной модели. "
            "Проверьте права доступа в Amvera или попробуйте выбрать другую модель.",
        )


def _extract_amvera_answer(data: dict[str, Any]) -> str:
    """Попытаться достать текст ответа из структуры ответа Amvera."""

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

    raise ValueError("Не удалось извлечь текст ответа из ответа модели")


def generate_response(context: str, question: str) -> str:
    """Запрос в Amvera GPT-модель + кэш Redis."""

    try:
        cache_key = hashlib.md5(f"{question}:{context}".encode()).hexdigest()
        cached = redis_client.get(cache_key)
        if cached:
            print("🎯 Ответ из кэша Redis")
            return cached

        normalized_token = _ensure_amvera_token()
        if not normalized_token:
            return ERROR_MESSAGE

        payload = _build_amvera_payload(AMVERA_GPT_MODEL, context, question)

        try:
            response = _perform_amvera_request(
                AMVERA_GPT_URL,
                normalized_token,
                payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            print(f"⚠️ Не удалось выполнить запрос к Amvera API: {exc}")
            return ERROR_MESSAGE

        if not response.ok:
            _log_amvera_error(response)
            return ERROR_MESSAGE

        try:
            data = response.json()
        except ValueError:
            print("⚠️ Не удалось распарсить ответ Amvera как JSON")
            return ERROR_MESSAGE

        try:
            answer = _extract_amvera_answer(data)
        except ValueError as exc:
            print(f"⚠️ {exc}")
            return ERROR_MESSAGE

        try:
            redis_client.setex(cache_key, 3600, answer)  # TTL 1 час
        except Exception as exc:  # pragma: no cover - сбой Redis не критичен для ответа
            print(f"⚠️ Не удалось сохранить ответ в Redis: {exc}")
        else:
            print("💾 Ответ сохранён в кэш Redis")
        return answer
    except Exception as exc:  # pragma: no cover - непредвиденные ошибки
        print(f"⚠️ Ошибка при обращении к модели: {exc}")
        return ERROR_MESSAGE

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    """Основной endpoint чата с гостями (RAG)."""
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()
    session_id = data.get("session_id") or hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()

    if not question:
        return jsonify({"response": "Пожалуйста, введите вопрос."})

    print(f"\n💬 Вопрос [{session_id[:8]}]: {question}")

    # 1) Команда сброса
    if question.lower() in {"отмена", "сброс", "начать заново", "стоп", "reset"}:
        redis_client.delete(f"booking_session:{session_id}")
        return jsonify({
            "response": "Диалог сброшен. Чем могу помочь?",
            "session_id": session_id
        })

    # 2) Диалог про бронирование/цены (если модуль вернул ответ — отдаем его)
    booking_result = handle_price_dialog(session_id, question, redis_client)
    if booking_result:
        return jsonify({
            "response": booking_result["answer"],
            "session_id": session_id,
            "mode": booking_result.get("mode", "booking")
        })

    # 3) Обычный RAG
    normalized = normalize_text(question)
    print(f"📝 Нормализовано: {normalized}")

    query_embedding = encode(normalized)
    print(f"🔢 Embedding размер запроса: {len(query_embedding)}")

    print("🔍 Поиск по коллекциям:", ", ".join(COLLECTIONS))
    all_results = search_all_collections(query_embedding, limit=5)

    if not all_results:
        print("❌ Ничего не найдено ни в одной коллекции")
        return jsonify({
            "response": "Извините, не нашёл информации по вашему вопросу. Попробуйте переформулировать или свяжитесь с администратором.",
            "session_id": session_id
        })

    print("\n📊 Топ-результаты:")
    for i, res in enumerate(all_results, 1):
        preview = res["text"][:100].replace("\n", " ")
        print(f"   {i}. [{res['collection']}] score={res['score']:.4f} | {preview}...")

    # Контекст из топ-3
    context = "\n\n".join([res["text"] for res in all_results[:3]]) or ""
    if not context.strip():
        print("⚠️ Контекст пуст после извлечения payload['text']")
        return jsonify({
            "response": "Извините, не удалось сформировать ответ. Попробуйте переформулировать вопрос.",
            "session_id": session_id
        })

    print(f"\n📄 Итоговый контекст ({len(context)} символов):\n{context[:300]}...\n")

    answer = generate_response(context, question)
    print(f"✅ Ответ сгенерирован: {answer[:100]}...\n")

    return jsonify({
        "response": answer,
        "session_id": session_id,
        "debug_info": {
            "top_collection": all_results[0]["collection"] if all_results else None,
            "top_score": all_results[0]["score"] if all_results else 0,
            "results_count": len(all_results),
            "embedding_dim": len(query_embedding)
        }
    })

@app.route("/api/debug/qdrant", methods=["GET"])
def debug_qdrant():
    """Проверка состояния Qdrant и списка коллекций."""
    try:
        collections = qdrant_client.get_collections().collections
        return jsonify({
            "status": "ok",
            "collections": [c.name for c in collections]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/debug/redis", methods=["GET"])
def debug_redis():
    """Проверка состояния Redis."""
    try:
        redis_client.ping()
        return jsonify({"status": "ok", "message": "Redis connection active"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/debug/search", methods=["POST"])
def debug_search():
    """Отладка поиска без генерации ответа."""
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "message required"}), 400

    normalized = normalize_text(question)
    vec = encode(normalized)
    results = search_all_collections(vec, limit=10)

    return jsonify({
        "question": question,
        "normalized": normalized,
        "embedding_dim": len(vec),
        "results": [
            {
                "collection": r["collection"],
                "score": r["score"],
                "text_preview": r["text"][:200]
            }
            for r in results
        ]
    })


# Позволяем обращаться как к `/api/debug/amvera`, так и к `/api/debug/amvera/`,
# чтобы избежать неожиданных 404 из‑за отличий в настройках прокси.
@app.route("/api/debug/amvera", methods=["GET"], strict_slashes=False)
def debug_amvera():
    """Проверка доступности Amvera GPT-модели."""

    prompt = request.args.get("prompt", "Привет! Ответь 'ok'.")
    model_name = request.args.get("model", AMVERA_GPT_MODEL).strip() or AMVERA_GPT_MODEL

    token = _ensure_amvera_token()
    if not token:
        return jsonify(
            {
                "status": "error",
                "message": "Не задан токен (AMVERA_GPT_TOKEN)",
            }
        ), 503

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "text": "Ты — простая проверка доступа к API.",
            },
            {
                "role": "user",
                "text": prompt,
            },
        ],
    }

    try:
        response = _perform_amvera_request(
            AMVERA_GPT_URL,
            token,
            payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Не удалось выполнить запрос: {exc}",
                }
            ),
            502,
        )

    if response.ok:
        try:
            response_json = response.json()
        except ValueError:
            response_json = {"raw": response.text}
        return jsonify(
            {
                "status": "ok",
                "model": model_name,
                "prompt": prompt,
                "response": response_json,
            }
        )

    _log_amvera_error(response)
    try:
        error_body = response.json()
    except ValueError:
        error_body = {"raw": response.text}
    return (
        jsonify(
            {
                "status": "error",
                "message": "Amvera API вернул ошибку",
                "http_status": response.status_code,
                "details": error_body,
            }
        ),
        response.status_code,
    )

@app.route("/api/debug/model", methods=["GET"])
def debug_model():
    """Информация о модели эмбеддингов."""
    return jsonify({
        "model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension()
    })

@app.route("/health")
def health():
    """Health check."""
    return "OK", 200

# ----------------------------
# ROOT PAGE
# ----------------------------
_DEFAULT_ENDPOINT_ORDER = [
    "/api/chat",
    "/api/debug/qdrant",
    "/api/debug/redis",
    "/api/debug/search",
    "/api/debug/amvera",
    "/api/debug/model",
    "/api/debug/status",
    "/health",
]


def _collect_public_endpoints() -> list[str]:
    """Собрать список зарегистрированных эндпоинтов без `static`."""

    collected: dict[str, None] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static" or rule.rule == "/":
            continue
        collected.setdefault(rule.rule, None)

    ordered: list[str] = []
    for path in _DEFAULT_ENDPOINT_ORDER:
        if path in collected:
            ordered.append(path)
            collected.pop(path)

    # Добавим любые дополнительные эндпоинты, не попавшие в предопределённый порядок.
    ordered.extend(sorted(collected.keys()))
    return ordered


@app.route("/")
def home():
    """Главная страница API."""
    return jsonify({
        "status": "ok",
        "message": "Усадьба 'Четыре Сезона' - AI Assistant",
        "version": "3.1",
        "features": ["RAG", "Booking Dialog", "Redis Cache"],
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "endpoints": _collect_public_endpoints(),
    })


@app.route("/api/debug/status", methods=["GET"])
def debug_status():
    """Aggregate status information about external dependencies."""

    services = {}

    # Qdrant status
    try:
        qdrant_client.get_collections()
        services["qdrant"] = {
            "status": "ok",
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
        }
    except Exception as exc:
        services["qdrant"] = {
            "status": "error",
            "message": str(exc),
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
        }

    # Redis status
    try:
        redis_client.ping()
        services["redis"] = {
            "status": "ok",
            "host": REDIS_HOST,
            "port": REDIS_PORT,
        }
    except Exception as exc:
        services["redis"] = {
            "status": "error",
            "message": str(exc),
            "host": REDIS_HOST,
            "port": REDIS_PORT,
        }

    # Embedding model status
    try:
        services["embedding_model"] = {
            "status": "ok",
            "name": EMBEDDING_MODEL_NAME,
            "dimension": model.get_sentence_embedding_dimension(),
        }
    except Exception as exc:
        services["embedding_model"] = {
            "status": "error",
            "name": EMBEDDING_MODEL_NAME,
            "message": str(exc),
        }

    if AMVERA_GPT_URL:
        services["amvera_gpt"] = {
            "status": "configured",
            "url": AMVERA_GPT_URL,
        }
    else:
        services["amvera_gpt"] = {
            "status": "not_configured",
        }

    overall_status = "ok" if all(s.get("status") != "error" for s in services.values()) else "degraded"

    return jsonify({
        "status": overall_status,
        "services": services,
    })

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    # Для локального запуска: host=0.0.0.0, порт можно переопределить через PORT
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

