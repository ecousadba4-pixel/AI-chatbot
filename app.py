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

def select_collection(query_embedding: list) -> str:
    """Автоматический выбор коллекции по наибольшей плотности похожести."""
    try:
        best_collection = None
        best_score = -1
        
        print("🔍 Поиск лучшей коллекции...")
        for coll in COLLECTIONS:
            search = qdrant_client.search(
                collection_name=coll,
                query_vector=query_embedding,
                limit=1
            )
            if search and len(search) > 0:
                score = search[0].score
                print(f"   📊 {coll}: score = {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_collection = coll
            else:
                print(f"   ⚠️ {coll}: нет результатов")
        
        result = best_collection or "hotel_info_v2"
        print(f"✅ Выбрана коллекция: {result} (score: {best_score:.4f})")
        return result
    except Exception as e:
        print(f"⚠️ Collection selection error: {e}")
        return "hotel_info_v2"

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
                all_results.append({
                    "collection": coll,
                    "score": hit.score,
                    "text": hit.payload.get("text", "")
                })
        except Exception as e:
            print(f"⚠️ Ошибка поиска в {coll}: {e}")
    
    # Сортируем по score (убывание)
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]

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
    
    print(f"\n💬 Вопрос [{session_id[:8]}]: {question}")
    
    # 1. Предобработка и векторизация
    normalized = normalize_text(question)
    print(f"📝 Нормализовано: {normalized}")
    
    # КРИТИЧНО: преобразуем numpy array в list
    query_embedding = model.encode(normalized).tolist()
    print(f"🔢 Embedding размер: {len(query_embedding)}")
    
    # 2. Поиск по всем коллекциям для лучших результатов
    print(f"🔍 Поиск по всем коллекциям...")
    all_results = search_all_collections(query_embedding, limit=5)
    
    if not all_results:
        print("❌ Ничего не найдено ни в одной коллекции!")
        return jsonify({
            "response": "Извините, не нашёл информации по вашему вопросу. Попробуйте переформулировать или свяжитесь с администратором.",
            "session_id": session_id
        })
    
    # 3. Формируем контекст из топ-результатов
    print(f"\n📊 Топ-5 результатов:")
    for i, res in enumerate(all_results, 1):
        print(f"   {i}. [{res['collection']}] score={res['score']:.4f} | text: {res['text'][:100]}...")
    
    context = "\n\n".join([res["text"] for res in all_results[:3]])
    
    if not context.strip():
        print("⚠️ Контекст пустой после извлечения text!")
        return jsonify({
            "response": "Извините, не удалось сформировать ответ. Попробуйте переформулировать вопрос.",
            "session_id": session_id
        })
    
    print(f"\n📄 Итоговый контекст ({len(context)} символов):\n{context[:300]}...\n")
    
    # 4. Генерация финального ответа (с кэшированием в Redis)
    answer = generate_response(context, question)
    print(f"✅ Ответ сгенерирован: {answer[:100]}...\n")
    
    return jsonify({
        "response": answer,
        "session_id": session_id,
        "debug_info": {
            "top_collection": all_results[0]["collection"] if all_results else None,
            "top_score": all_results[0]["score"] if all_results else 0,
            "results_count": len(all_results)
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
    """Отладочный endpoint для тестирования поиска."""
    data = request.get_json()
    question = data.get("message", "").strip()
    
    if not question:
        return jsonify({"error": "message required"})
    
    normalized = normalize_text(question)
    query_embedding = model.encode(normalized).tolist()
    
    results = search_all_collections(query_embedding, limit=10)
    
    return jsonify({
        "question": question,
        "normalized": normalized,
        "results": [
            {
                "collection": r["collection"],
                "score": r["score"],
                "text_preview": r["text"][:200]
            }
            for r in results
        ]
    })

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
        "version": "2.0",
        "endpoints": [
            "/api/chat",
            "/api/debug/qdrant",
            "/api/debug/redis",
            "/api/debug/search",
            "/health"
        ]
    })

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
