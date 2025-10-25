import os
import re
import hashlib
import requests
import pymorphy3
import numpy as np
import redis
from flask import Flask, request, jsonify
from flask_cors import CORS
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from datetime import datetime

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
print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
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
    """Отправка запроса в Amvera GPT‑модель для генерации ответа с кэшированием в Redis."""
    try:
        # Проверка кэша Redis
        cache_key = hashlib.md5(f"{question}:{context}".encode()).hexdigest()
        cached = redis_client.get(cache_key)
        if cached:
            print(f"🎯 Ответ из кэша Redis")
            return cached
        
        # Генерация нового ответа
        headers = {
            "Authorization": f"Bearer {AMVERA_GPT_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt",
            "messages": [
                {
                    "role": "system", 
                    "content": "Ты — ассистент загородного отеля усадьбы 'Четыре Сезона'. Отвечай гостям кратко, дружелюбно и только на основе предоставленной информации. Если информации нет в контексте, вежливо скажи об этом."
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
            # Сохранение в Redis кэш (TTL 1 час = 3600 секунд)
            redis_client.setex(cache_key, 3600, answer)
            print(f"💾 Ответ сохранён в кэш Redis")
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
    """Основной endpoint для чата с гостями."""
    data = request.get_json()
    question = data.get("message", "").strip()
    session_id = data.get("session_id") or hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()
    
    if not question:
        return jsonify({"response": "Пожалуйста, введите вопрос."})
    
    print(f"💬 Вопрос [{session_id[:8]}]: {question}")
    
    # 1. Предобработка и векторизация
    normalized = normalize_text(question)
    query_embedding = model.encode(normalized)
    
    # 2. Определяем подходящую коллекцию
    collection = select_collection(query_embedding)
    print(f"🎯 Коллекция выбрана: {collection}")
    
    # 3. Ищем релевантные документы в Qdrant
    search_results = qdrant_client.search(
        collection_name=collection,
        query_vector=query_embedding,
        limit=3
    )
    
    # 4. Формируем контекст из найденных документов
    context = "\n".join([hit.payload.get("text", "") for hit in search_results])
    if not context:
        context = "Информация временно недоступна."
    
    print(f"📄 Найденный контекст (первые 200 символов): {context[:200]}...")
    
    # 5. Генерация финального ответа (с кэшированием в Redis)
    answer = generate_response(context, question)
    
    return jsonify({
        "response": answer,
        "collection": collection,
        "session_id": session_id
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

@app.route("/health")
def health():
    """Health check для Amvera и мониторинга."""
    return "OK", 200

@app.route("/")
def home():
    """Главная страница API."""
    return jsonify({
        "status": "ok",
        "message": "Усадьба 'Четыре Сезона' - AI Assistant",
        "version": "1.0",
        "endpoints": [
            "/api/chat",
            "/api/debug/qdrant",
            "/api/debug/redis",
            "/health"
        ]
    })

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
