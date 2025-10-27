"""REST API чат-бота для усадьбы "Четыре сезона"."""
from __future__ import annotations

import os
from typing import Any

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from amvera import (
    AmveraError,
    build_payload,
    cache_key,
    ensure_token,
    extract_answer,
    log_error,
    perform_request,
)
from config import Settings
from price_dialog import handle_price_dialog
from rag import SearchResult, encode, normalize_text, search_all_collections
from services import Dependencies, create_dependencies

# ----------------------------
# ИНИЦИАЛИЗАЦИЯ
# ----------------------------
settings = Settings.from_env()
deps: Dependencies = create_dependencies(settings)

app = Flask(__name__)
CORS(app)

CANCEL_COMMANDS = {"отмена", "сброс", "начать заново", "стоп", "reset"}
ERROR_MESSAGE = "Извините, не удалось получить ответ. Пожалуйста, попробуйте позже."
DEFAULT_COLLECTIONS = list(settings.default_collections)

# ----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------------
def _log_startup_information() -> None:
    print(
        "✅ Connected to Qdrant at "
        f"{settings.qdrant_host}:{settings.qdrant_port} (https={settings.qdrant_https})"
    )
    print(f"✅ Connected to Redis at {settings.redis_host}:{settings.redis_port}")
    print(
        "🔢 Embedding dimension:",
        deps.embedding_model.get_sentence_embedding_dimension(),
    )
    print(
        f"🤖 Amvera GPT endpoint: {settings.amvera_url} (model={settings.amvera_model})"
    )


def _filter_existing_collections(
    client, requested: list[str], fallback: list[str]
) -> list[str]:
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

    fallback_candidates = [name for name in fallback if name in available] or sorted(
        available
    )
    print(
        "⚠️ Ни одна из запрошенных коллекций не найдена. Используем fallback: "
        + ", ".join(fallback_candidates)
    )
    return fallback_candidates


COLLECTIONS = _filter_existing_collections(
    deps.qdrant,
    DEFAULT_COLLECTIONS.copy(),
    DEFAULT_COLLECTIONS,
)


def _collect_public_endpoints() -> list[str]:
    collected: dict[str, None] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static" or rule.rule == "/":
            continue
        collected.setdefault(rule.rule, None)

    default_order = [
        "/api/chat",
        "/api/debug/qdrant",
        "/api/debug/redis",
        "/api/debug/search",
        "/api/debug/amvera",
        "/api/debug/model",
        "/api/debug/status",
        "/health",
    ]

    ordered: list[str] = []
    for path in default_order:
        if path in collected:
            ordered.append(path)
            collected.pop(path)

    ordered.extend(sorted(collected.keys()))
    return ordered


# ----------------------------
# ОСНОВНАЯ ЛОГИКА
# ----------------------------
def _json_reply(session_id: str, message: str, **extra: Any):
    payload = {"response": message, "session_id": session_id}
    payload.update(extra)
    return jsonify(payload)


def _build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(result.text for result in results)


def _generate_response(context: str, question: str) -> str:
    redis_key = cache_key(context, question)

    cached = deps.redis.get(redis_key)
    if cached:
        print("🎯 Ответ из кэша Redis")
        return cached

    try:
        token = ensure_token(settings)
    except AmveraError as exc:
        print(f"⚠️ {exc}")
        return ERROR_MESSAGE

    payload = build_payload(settings.amvera_model, context, question)

    try:
        response = perform_request(settings, token, payload, timeout=60)
    except requests.RequestException as exc:
        print(f"⚠️ Не удалось выполнить запрос к Amvera API: {exc}")
        return ERROR_MESSAGE

    if not response.ok:
        log_error(response)
        return ERROR_MESSAGE

    try:
        data = response.json()
    except ValueError:
        print("⚠️ Не удалось распарсить ответ Amvera как JSON")
        return ERROR_MESSAGE

    try:
        answer = extract_answer(data)
    except AmveraError as exc:
        print(f"⚠️ {exc}")
        return ERROR_MESSAGE

    try:
        deps.redis.setex(redis_key, 3600, answer)
    except Exception as exc:  # pragma: no cover - сбой Redis не критичен
        print(f"⚠️ Не удалось сохранить ответ в Redis: {exc}")
    else:
        print("💾 Ответ сохранён в кэш Redis")

    return answer


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat() -> Any:
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()
    session_id = data.get("session_id") or os.urandom(16).hex()

    if not question:
        return _json_reply(session_id, "Пожалуйста, введите вопрос.")

    print(f"\n💬 Вопрос [{session_id[:8]}]: {question}")

    if question.lower() in CANCEL_COMMANDS:
        deps.redis.delete(f"booking_session:{session_id}")
        return _json_reply(session_id, "Диалог сброшен. Чем могу помочь?")

    booking_result = handle_price_dialog(
        session_id,
        question,
        deps.redis,
        deps.morph,
    )
    if booking_result:
        return _json_reply(
            session_id,
            booking_result["answer"],
            mode=booking_result.get("mode", "booking"),
        )

    normalized = normalize_text(question, deps.morph)
    print(f"📝 Нормализовано: {normalized}")

    query_embedding = encode(normalized, deps.embedding_model)
    print(f"🔢 Embedding размер запроса: {len(query_embedding)}")

    print("🔍 Поиск по коллекциям:", ", ".join(COLLECTIONS))
    search_results = search_all_collections(
        deps.qdrant,
        COLLECTIONS,
        query_embedding,
        limit=5,
    )

    if not search_results:
        print("❌ Ничего не найдено ни в одной коллекции")
        return _json_reply(
            session_id,
            "Извините, не нашёл информации по вашему вопросу. "
            "Попробуйте переформулировать или свяжитесь с администратором.",
        )

    print("\n📊 Топ-результаты:")
    for index, result in enumerate(search_results, start=1):
        preview = result.text[:100].replace("\n", " ")
        print(
            f"   {index}. [{result.collection}] score={result.score:.4f} | {preview}..."
        )

    context = _build_context(search_results[:3])
    if not context.strip():
        print("⚠️ Контекст пуст после извлечения payload['text']")
        return _json_reply(
            session_id,
            "Извините, не удалось сформировать ответ. "
            "Попробуйте переформулировать вопрос.",
        )

    print(f"\n📄 Итоговый контекст ({len(context)} символов):\n{context[:300]}...\n")

    answer = _generate_response(context, question)
    print(f"✅ Ответ сгенерирован: {answer[:100]}...\n")

    return _json_reply(
        session_id,
        answer,
        debug_info={
            "top_collection": search_results[0].collection,
            "top_score": search_results[0].score,
            "results_count": len(search_results),
            "embedding_dim": len(query_embedding),
        },
    )


@app.route("/api/debug/qdrant", methods=["GET"])
def debug_qdrant() -> Any:
    try:
        collections = deps.qdrant.get_collections().collections
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)})

    return jsonify({
        "status": "ok",
        "collections": [collection.name for collection in collections],
    })


@app.route("/api/debug/redis", methods=["GET"])
def debug_redis() -> Any:
    try:
        deps.redis.ping()
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)})

    return jsonify({"status": "ok", "message": "Redis connection active"})


@app.route("/api/debug/search", methods=["POST"])
def debug_search() -> Any:
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"error": "message required"}), 400

    normalized = normalize_text(question, deps.morph)
    vector = encode(normalized, deps.embedding_model)
    results = search_all_collections(
        deps.qdrant,
        COLLECTIONS,
        vector,
        limit=10,
    )

    return jsonify(
        {
            "question": question,
            "normalized": normalized,
            "embedding_dim": len(vector),
            "results": [
                {
                    "collection": result.collection,
                    "score": result.score,
                    "text_preview": result.text[:200],
                }
                for result in results
            ],
        }
    )


@app.route("/api/debug/amvera", methods=["GET"], strict_slashes=False)
def debug_amvera() -> Any:
    prompt = request.args.get("prompt", "Привет! Ответь 'ok'.")
    model_name = request.args.get("model") or settings.amvera_model

    try:
        token = ensure_token(settings)
    except AmveraError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 503

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "text": "Ты — простая проверка доступа к API."},
            {"role": "user", "text": prompt},
        ],
    }

    try:
        response = perform_request(settings, token, payload, timeout=30)
    except requests.RequestException as exc:
        return jsonify({"status": "error", "message": f"Не удалось выполнить запрос: {exc}"}), 502

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

    log_error(response)
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
def debug_model() -> Any:
    return jsonify(
        {
            "model": settings.embedding_model,
            "embedding_dimension": deps.embedding_model.get_sentence_embedding_dimension(),
        }
    )


@app.route("/api/debug/status", methods=["GET"])
def debug_status() -> Any:
    services: dict[str, Any] = {}

    try:
        deps.qdrant.get_collections()
        services["qdrant"] = {
            "status": "ok",
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
        }
    except Exception as exc:
        services["qdrant"] = {
            "status": "error",
            "message": str(exc),
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
        }

    try:
        deps.redis.ping()
        services["redis"] = {
            "status": "ok",
            "host": settings.redis_host,
            "port": settings.redis_port,
        }
    except Exception as exc:
        services["redis"] = {
            "status": "error",
            "message": str(exc),
            "host": settings.redis_host,
            "port": settings.redis_port,
        }

    try:
        services["embedding_model"] = {
            "status": "ok",
            "name": settings.embedding_model,
            "dimension": deps.embedding_model.get_sentence_embedding_dimension(),
        }
    except Exception as exc:
        services["embedding_model"] = {
            "status": "error",
            "name": settings.embedding_model,
            "message": str(exc),
        }

    services["amvera_gpt"] = {
        "status": "configured" if settings.amvera_url else "not_configured",
        "url": settings.amvera_url,
    }

    overall_status = (
        "ok" if all(service.get("status") != "error" for service in services.values()) else "degraded"
    )

    return jsonify({"status": overall_status, "services": services})


@app.route("/health")
def health() -> Any:
    return "OK", 200


@app.route("/")
def home() -> Any:
    return jsonify(
        {
            "status": "ok",
            "message": "Усадьба 'Четыре Сезона' - AI Assistant",
            "version": "4.0",
            "features": ["RAG", "Booking Dialog", "Redis Cache"],
            "embedding_model": settings.embedding_model,
            "embedding_dim": deps.embedding_model.get_sentence_embedding_dimension(),
            "endpoints": _collect_public_endpoints(),
        }
    )


# ----------------------------
# ENTRY POINT
# ----------------------------
_log_startup_information()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
