from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import redis
import hashlib
import json
import re
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from price_dialog import handle_price_dialog
from session_manager import get_recent_messages, save_message

app = Flask(__name__)
CORS(app)

# =================== CONFIG AND CONNECTIONS ===================
QDRANT_HOST = os.getenv("QDRANT_HOST", "amvera-karinausadba-run-u4s-ai-chatbot")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL", "")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "")
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
embedding_model = SentenceTransformer("sergeyzh/rubert-mini-frida")
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, https=False, api_key=QDRANT_API_KEY)

# =================== EMBEDDING CENTROIDS (для адаптера) ===================
COLLECTION_EMBEDDINGS = {
    "hotel_rooms_v2": embedding_model.encode("забронировать номер, категория, удобства, вместимость, кровати, домик").tolist(),
    "hotel_info_v2": embedding_model.encode("территория отеля, услуги, питание, ресторан, условия проживания").tolist(),
    "hotel_support_v2": embedding_model.encode("контакт, адрес, правила заселения, звонок, как доехать, частые вопросы").tolist(),
}

# =================== HELPER FUNCTIONS ===================
def normalize_query(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zа-я0-9ё\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def auto_detect_collection(vec):
    """Определяет, к какой тематике вопрос ближе по embedding-схожести."""
    similarities = {name: cosine_similarity(vec, center) for name, center in COLLECTION_EMBEDDINGS.items()}
    best = max(similarities, key=similarities.get)
    confidence = similarities[best]
    if confidence < 0.45:
        return None
    return best

def get_context_from_qdrant(query, top_n=6):
    try:
        query_norm = normalize_query(query)
        vec = embedding_model.encode(query_norm).tolist()

        detected_collection = auto_detect_collection(vec)
        collections = ["hotel_rooms_v2", "hotel_info_v2", "hotel_support_v2"]
        hits = []

        if detected_collection:
            main_results = qdrant_client.search(collection_name=detected_collection, query_vector=vec, limit=top_n)
            hits.extend(main_results)

        # Fallback: если уверенность низкая — комбинированный поиск
        if not detected_collection or len(hits) < top_n // 2:
            for col in collections:
                if col == detected_collection:
                    continue
                results = qdrant_client.search(collection_name=col, query_vector=vec, limit=2)
                hits.extend(results)

        hits = sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]
        context_blocks = []
        for h in hits:
            txt = h.payload.get("text", "").strip()
            src = h.payload.get("metadata", {}).get("source", "")
            if txt:
                context_blocks.append(f"[{src}]: {txt}")

        return "\n\n".join(context_blocks) if context_blocks else "Нет подходящей информации."
    except Exception as e:
        return f"⚠️ Ошибка Qdrant: {str(e)}"

# =================== GPT AND CACHE ===================
def make_cache_key(question, context):
    raw = f"{question}|{context}"
    return hashlib.sha256(raw.encode()).hexdigest()

def get_gpt_response_cached(question, context):
    key = make_cache_key(question, context)
    cached = redis_client.get(key)
    if cached:
        return cached
    answer = generate_gpt_answer(question, context)
    redis_client.setex(key, 259200, answer)  # кеш 3 дня
    return answer

def generate_gpt_answer(question, context):
    payload = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "text": "Ты — дружелюбный чат-бот загородного эко-отеля. Отвечай емко и естественно, используя только контекст. Если информации нет — честно напиши, что она отсутствует."},
            {"role": "user", "text": f"Контекст:\n{context}\n\nВопрос: {question}"}
        ]
    }
    headers = {"X-Auth-Token": f"Bearer {AMVERA_GPT_TOKEN}", "Content-Type": "application/json"}

    try:
        r = requests.post(AMVERA_GPT_URL, headers=headers, json=payload, timeout=40)
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Ответ не получен.")
    except Exception as e:
        return f"Ошибка GPT: {e}"

# =================== LOGIC ===================
def is_booking_related(text):
    t = text.lower()
    return any(k in t for k in ["забронировать", "цена", "категория", "номер", "вместимость", "бронь"])

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    session_id = body.get("session_id", "anon")

    if not question:
        return jsonify({"answer": "Пожалуйста, задайте вопрос."}), 200

    if is_booking_related(question):
        dlg = handle_price_dialog(session_id, question)
        if dlg:
            return jsonify(dlg), 200

    context_text = get_context_from_qdrant(question, top_n=6)
    history = get_recent_messages(session_id, limit=5)
    history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history]) if history else ""

    answer = get_gpt_response_cached(question, f"{history_text}\n{context_text}")
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return jsonify({"answer": answer, "contextSource": "rag+adaptive"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
