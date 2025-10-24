import os
import re
import json
import redis
import hashlib
import requests
import pymorphy3
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from datetime import datetime
from session_manager import get_recent_messages, save_message

# ----------------------------
# INIT
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# ENV VARIABLES
# ----------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "amvera-karinausadba-run-u4s-ai-chatbot")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
COLLECTIONS = ["hotel_info_v2", "hotel_rooms_v2", "hotel_support_v2"]

# ----------------------------
# CONNECTIONS
# ----------------------------
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True
)

morph = pymorphy3.MorphAnalyzer()
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    https=False  # обязательная настройка для внутреннего домена Amvera
)

print(f"✅ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
print("🧠 Embedding model loaded")

# ----------------------------
# FUNCTIONS
# ----------------------------

def normalize_text(text: str) -> str:
    """Базовая лемматизация вопроса."""
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)

def select_collection(query_embedding: np.ndarray) -> str:
    """Автоматический выбор коллекции по наибольшей плотности похожести."""
    try:
        best_collection = None
        best_score = -1
        for coll in COLLECTIONS:
            search = qdrant_client.search(
                collection_name=coll,
                query_vector=query_embedding,
                limit=1
            )
            if search and search[0].score > best_score:
                best_score = search[0].score
                best_collection = coll
        return best_collection or "hotel_info_v2"
    except Exception as e:
        print(f"⚠️ Collection selection error: {e}")
        return "hotel_info_v2"

def generate_response(context: str, question: str) -> str:
    """Отправка запроса в Amvera GPT‑модель для генерации ответа."""
    try:
        headers = {
            "Authorization": f"Bearer {AMVERA_GPT_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt",
            "messages": [
                {"role": "system", "content": "Ты — ассистент отеля усадьбы, отвечай гостям кратко и дружелюбно."},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос гостя: {question}"}
            ]
        }
        r = requests.post(AMVERA_GPT_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"Ошибка генерации ответа: {r.text}"
    except Exception as e:
        return f"⚠️ Ошибка при обращении к модели: {e}"

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()
    session_id = data.get("session_id") or hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()

    if not question:
        return jsonify({"response": "Пожалуйста, введите вопрос."})

    print(f"💬 Вопрос: {question}")

    # 1. Предобработка и векторизация
    normalized = normalize_text(question)
    query_embedding = model.encode(normalized)

    # 2. Определяем подходящую коллекцию
    collection = select_collection(query_embedding)
    print(f"🎯 Коллекция выбрана: {collection}")

    # 3. Ищем релевантные документы
    search_results = qdrant_client.search(
        collection_name=collection,
        query_vector=query_embedding,
        limit=3
    )
    context = "\n".join([hit.payload.get("text", "") for hit in search_results])
    if not context:
        context = "Информация временно недоступна. Пожалуйста, уточните чуть позже."

    # 4. Генерация финального ответа
    answer = generate_response(context, question)

    # 5. Сохранение в историю Redis/Qdrant
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    return jsonify({"response": answer, "collection": collection, "session_id": session_id})


@app.route("/api/debug/qdrant", methods=["GET"])
def debug_qdrant():
    """Проверка состояния Qdrant и списков коллекций."""
    try:
        collections = qdrant_client.get_collections().collections
        return jsonify({
            "status": "ok",
            "collections": [c.name for c in collections]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Hotel assistant online"})


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

