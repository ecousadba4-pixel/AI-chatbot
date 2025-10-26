import os
import re
import hashlib
import requests
import pymorphy3
import numpy as np
import redis
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datetime import datetime

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

# Сопоставление старых названий коллекций новым. Помогает пережить переименования
# без необходимости немедленно менять переменные окружения.
COLLECTION_ALIASES: dict[str, str] = {
    "hotel_ru": "hotel_knowledge",
}


def _normalize_collection_names(names: list[str]) -> list[str]:
    """Применяем алиасы и удаляем дубликаты, сохраняя порядок."""

    normalized: list[str] = []
    seen: set[str] = set()

    for name in names:
        target = COLLECTION_ALIASES.get(name, name)
        if target not in seen:
            seen.add(target)
            normalized.append(target)
    return normalized
QDRANT_HOST = os.getenv("QDRANT_HOST", "amvera-karinausadba-run-u4s-ai-chatbot")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_HTTPS = os.getenv("QDRANT_HTTPS", "false").lower() in ("1", "true", "yes")

AMVERA_GPT_URL = os.getenv(
    "AMVERA_GPT_URL", "https://kong-proxy.yc.amvera.ru/api/v1/models/gpt"
)
AMVERA_GPT_MODEL = os.getenv("AMVERA_GPT_MODEL", "gpt-5").strip() or "gpt-5"
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN")
AMVERA_AUTH_HEADER = os.getenv("AMVERA_AUTH_HEADER", "X-Auth-Token")
AMVERA_AUTH_PREFIX = os.getenv("AMVERA_AUTH_PREFIX", "Bearer")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Список коллекций, по которым ищем
def _load_collections() -> list[str]:
    """Определение списка коллекций для поиска с учётом переменных окружения."""

    env_json = os.getenv("COLLECTIONS_JSON")
    if env_json:
        try:
            parsed = json.loads(env_json)
        except json.JSONDecodeError:
            print(
                "⚠️ COLLECTIONS_JSON не удалось распарсить как JSON — будет использован fallback",
                f"(значение: {env_json!r})",
            )
        else:
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                normalized = _normalize_collection_names(parsed)
                print(
                    "🔧 Источник коллекций: COLLECTIONS_JSON →",
                    ", ".join(normalized) or "<пусто>",
                )
                return normalized
            print(
                "⚠️ COLLECTIONS_JSON должно быть списком строк — будет использован fallback",
                f"(значение: {parsed!r})",
            )

    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    collection_name = os.getenv("COLLECTION_NAME")
    single_collection = qdrant_collection or collection_name
    if single_collection:
        source = "QDRANT_COLLECTION" if qdrant_collection else "COLLECTION_NAME"
        normalized = _normalize_collection_names([single_collection])
        print(
            f"🔧 Источник коллекций: {source} →",
            ", ".join(normalized) or "<пусто>",
        )
        return normalized

    comma_separated = os.getenv("COLLECTIONS")
    if comma_separated:
        items = [item.strip() for item in comma_separated.split(",") if item.strip()]
        if items:
            normalized = _normalize_collection_names(items)
            print(
                "🔧 Источник коллекций: COLLECTIONS →",
                ", ".join(normalized) or "<пусто>",
            )
            return normalized

    # Значения по умолчанию (актуальная единая коллекция)
    normalized = _normalize_collection_names(DEFAULT_COLLECTIONS)
    print(
        "🔧 Источник коллекций: значение по умолчанию →",
        ", ".join(normalized) or "<пусто>",
    )
    return normalized


COLLECTIONS = _load_collections()


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
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
)

morph = pymorphy3.MorphAnalyzer()

# >>> ГЛАВНОЕ ИЗМЕНЕНИЕ: русская крупная модель эмбеддингов <<<
# Размер эмбеддинга = 1024
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sberbank-ai/sbert_large_nlu_ru")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
ALLOW_EMBEDDING_DOWNLOAD = os.getenv("ALLOW_EMBEDDING_DOWNLOAD", "false").lower() in ("1", "true", "yes")


model = resolve_embedding_model(
    model_name=EMBEDDING_MODEL_NAME,
    candidate_paths=[
        EMBEDDING_MODEL_PATH,
        "/app/data/sberbank-ai/sbert_large_nlu_ru",
        "/data/sberbank-ai/sbert_large_nlu_ru",
    ],
    allow_download=ALLOW_EMBEDDING_DOWNLOAD,
)

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    https=QDRANT_HTTPS  # по умолчанию false для внутреннего домена; можно включить через переменные
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

def search_all_collections(query_embedding: list, limit: int = 5) -> list:
    """Поиск по всем коллекциям с объединением результатов."""
    all_results = []
    for coll in COLLECTIONS:
        try:
            search = qdrant_client.search(
                collection_name=coll,
                query_vector=query_embedding,
                limit=limit
            )
            for hit in search:
                payload = hit.payload or {}
                text = payload.get("text") or payload.get("text_bm25") or ""
                if not text and isinstance(payload.get("raw"), dict):
                    # Попробуем собрать fallback из известных полей.
                    raw = payload["raw"]
                    text_blocks = raw.get("text_blocks")
                    if isinstance(text_blocks, dict):
                        text = "\n".join(str(v) for v in text_blocks.values() if v)
                    if not text:
                        text = raw.get("text", "")
                    if not text and raw.get("category") == "faq":
                        q = raw.get("question")
                        a = raw.get("answer")
                        if q or a:
                            text = "Вопрос: {}\nОтвет: {}".format(q or "", a or "")

                all_results.append({
                    "collection": coll,
                    "score": hit.score,
                    "text": text
                })
        except Exception as e:
            print(f"⚠️ Ошибка поиска в {coll}: {e}")
    # Сортируем по score (убывание)
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]

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
            return "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."

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
            return "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."

        if response.ok:
            data = response.json()
            answer = None

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message") or {}
                    if isinstance(message, dict):
                        answer = (
                            message.get("content")
                            or message.get("text")
                        )

            if not answer:
                answer = data.get("output_text") or data.get("text")

            if not answer:
                raise ValueError("Не удалось извлечь текст ответа из ответа модели")
            redis_client.setex(cache_key, 3600, answer)  # TTL 1 час
            print("💾 Ответ сохранён в кэш Redis")
            return answer

        _log_amvera_error(response)
        return "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."
    except Exception as e:
        print(f"⚠️ Ошибка при обращении к модели: {e}")
        return "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."

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

