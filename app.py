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
    "hotel_info_v2": "hotel_knowledge",
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

AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN")

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
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return _normalize_collection_names(parsed)
        except json.JSONDecodeError:
            print("⚠️ COLLECTIONS_JSON не удалось распарсить как JSON — будет использован fallback")

    single_collection = (
        os.getenv("QDRANT_COLLECTION")
        or os.getenv("COLLECTION_NAME")
    )
    if single_collection:
        return _normalize_collection_names([single_collection])

    comma_separated = os.getenv("COLLECTIONS")
    if comma_separated:
        items = [item.strip() for item in comma_separated.split(",") if item.strip()]
        if items:
            return _normalize_collection_names(items)

    # Значения по умолчанию (актуальная единая коллекция)
    return _normalize_collection_names(DEFAULT_COLLECTIONS)


COLLECTIONS = _load_collections()

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

def generate_response(context: str, question: str) -> str:
    """Запрос в Amvera GPT-модель + кэш Redis."""
    try:
        cache_key = hashlib.md5(f"{question}:{context}".encode()).hexdigest()
        cached = redis_client.get(cache_key)
        if cached:
            print("🎯 Ответ из кэша Redis")
            return cached

        headers = {
            "Authorization": f"Bearer {AMVERA_GPT_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты — ассистент загородного отеля усадьбы 'Четыре Сезона'. "
                        "Отвечай гостям кратко, дружелюбно и только на основе предоставленной информации. "
                        "Если информации нет в контексте, вежливо скажи об этом."
                    )
                },
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос гостя: {question}"
                }
            ]
        }

        r = requests.post(AMVERA_GPT_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            answer = r.json()["choices"][0]["message"]["content"]
            redis_client.setex(cache_key, 3600, answer)  # TTL 1 час
            print("💾 Ответ сохранён в кэш Redis")
            return answer
        else:
            print(f"⚠️ Ошибка GPT API: {r.status_code} - {r.text}")
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
        "endpoints": [
            "/api/chat",
            "/api/debug/qdrant",
            "/api/debug/redis",
            "/api/debug/search",
            "/api/debug/model",
            "/api/debug/status",
            "/health"
        ]
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

