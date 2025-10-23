from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import redis
import hashlib
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json
import time

from price_dialog import handle_price_dialog  # твой модуль работы с Shelter Cloud

app = Flask(__name__)
CORS(app)

# ==== ENVIRONMENT VARIABLES ====
QDRANT_HOST = os.getenv("QDRANT_HOST", "amvera-karinausadba-run-u4s-ai-chatbot")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))   # порт Qdrant внутри Amvera — всегда 6333!
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hotel_docs")

AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL", "")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# Не прописывай password, если REDIS_PASSWORD не задан!
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

# ================= REDIS (3 дня) =================
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD if REDIS_PASSWORD else None,
    decode_responses=True
)

def make_gpt_cache_key(prompt, context):
    raw = f"{prompt}|{context}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"gpt:{key}"

def cached_gpt_response(prompt, context, token, expire=259200):  # 3 дня
    cache_key = make_gpt_cache_key(prompt, context)
    cached = redis_client.get(cache_key)
    if cached:
        return cached
    answer = amvera_gpt_query(prompt, context, token)
    redis_client.setex(cache_key, expire, answer)
    return answer

# ================= QDRANT =======================
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=False,  # Внутренний адрес — только http!
    api_key=QDRANT_API_KEY
)
embedding_model = SentenceTransformer("sergeyzh/rubert-mini-frida")

def get_context_from_qdrant(query, top_n=5):
    try:
        t0 = time.time()
        vec = embedding_model.encode(query).tolist()
        print("Embedding time:", time.time()-t0)
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_n
        )
        context = " ".join(hit.payload.get("text", "") for hit in results)
        return context if context.strip() else "Нет подходящих документов."
    except Exception as e:
        return f"Ошибка при получении контекста от Qdrant: {str(e)}"

# ================= GPT ==========================
def amvera_gpt_query(user_question, context, token):
    payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты чат-бот отеля. "
                    "Отвечай кратко и только по контексту. "
                    "Если данных нет — сообщай, что информации нет."
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
        return f"Ошибка GPT API: {str(e)}"

# ====== Улучшенная логика определения сценариев ======
def is_booking_request(text):
    booking_triggers = [
        "забронировать номер",
        "бронь номера",
        "бронирование номера",
        "снять номер",
        "освободить номер"
    ]
    ignore_phrases = [
        "еда", "доставка", "завтрак", "ужин", "обед", "заказать еду", "меню", "напитки",
        "room service", "доставка еды", "ужин в номер", "заказать питьевую воду", "чайник"
    ]
    lower = text.lower()
    if any(p in lower for p in ignore_phrases):
        return False
    return any(p in lower for p in booking_triggers)

# ================= ROUTES =======================
@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    user_input = payload.get("question", "").strip()
    user_id = payload.get("session_id", "anon")

    if not user_input:
        return jsonify({"answer": "Пожалуйста, задайте вопрос."}), 200

    if is_booking_request(user_input):
        dialog_response = handle_price_dialog(user_id, user_input)
        if dialog_response:
            return jsonify(dialog_response), 200

    context = get_context_from_qdrant(user_input, top_n=5)
    answer = cached_gpt_response(user_input, context, AMVERA_GPT_TOKEN)
    return jsonify({"answer": answer, "mode": "info"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
