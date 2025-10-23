from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import redis
import hashlib
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from price_dialog import handle_price_dialog  # модуль диалога с Shelter API

app = Flask(__name__)
CORS(app)

# ================= ENV ========================
QDRANT_HOST = os.getenv("QDRANT_HOST", "u4s-ai-chatbot-karinausadba.amvera.io")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 443))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hotel_docs")

AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL", "")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# ================= REDIS =======================
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

def make_gpt_cache_key(prompt, context):
    raw = f"{prompt}|{context}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"gpt:{key}"

def cached_gpt_response(prompt, context, token, expire=259200):  # 3 дня (в секундах)
    cache_key = make_gpt_cache_key(prompt, context)
    cached = redis_client.get(cache_key)
    if cached:
        return cached
    answer = amvera_gpt_query(prompt, context, token)
    redis_client.setex(cache_key, expire, answer)
    return answer

# ================= Qdrant ======================
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=True,
    api_key=QDRANT_API_KEY
)

embedding_model = SentenceTransformer("sergeyzh/rubert-mini-frida")

def get_context_from_qdrant(query, top_n=5):
    try:
        vec = embedding_model.encode(query).tolist()
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_n
        )
        context = " ".join(hit.payload.get("text", "") for hit in results)
        return context or "Нет подходящих документов."
    except Exception as e:
        return f"Ошибка при получении контекста от Qdrant: {str(e)}"

# ================= GPT =========================
def amvera_gpt_query(user_question, context, token):
    payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты чат-бот отеля. "
                    "Отвечай кратко и по сути на основе контекста. "
                    "Если ответа нет — сообщай, что данных нет."
                ),
            },
            {
                "role": "user",
                "text": f"Контекст:\n{context}\n\nВопрос: {user_question}",
            },
        ],
    }
    headers = {
        "X-Auth-Token": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(AMVERA_GPT_URL, headers=headers, json=payload, timeout=40)
        data = response.json() if response.content else {}
        answer = (
            data.get("choices", [{}])[0].get("message", {}).get("content")
            if data.get("choices") else None
        ) or data.get("content") or "Ответ не получен."
        return answer
    except Exception as e:
        return f"Ошибка при обращении к Amvera GPT API: {str(e)}"

# ================= ROUTES ======================
@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    user_input = payload.get("question", "").strip()
    user_id = payload.get("session_id", "anon")

    if not user_input:
        return jsonify({"answer": "Пожалуйста, введите вопрос."}), 200

    # 1. Проверяем, не сценарий ли это бронирования (Shelter)
    dialog_response = handle_price_dialog(user_id, user_input)
    if dialog_response:
        return jsonify(dialog_response), 200

    # 2. GPT-логика с контекстом и кэшированием
    context = get_context_from_qdrant(user_input, top_n=5)
    answer = cached_gpt_response(user_input, context, AMVERA_GPT_TOKEN)
    return jsonify({"answer": answer, "mode": "info"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
