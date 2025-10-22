from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Настройки Qdrant
QDRANT_HOST = "u4s-ai-chatbot-karinausadba.amvera.io"
QDRANT_PORT = 443
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "ВАШ_КЛЮЧ_ПО_УМОЛЧАНИЮ")
COLLECTION_NAME = "hotel_docs"

# Настройки Amvera GPT-5
AMVERA_GPT_URL = "https://kong-proxy.yc.amvera.ru/api/v1/models/gpt"
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "ВАШ_ТОКЕН_ПО_УМОЛЧАНИЮ")

# Инициализация Flask
app = Flask(__name__)
CORS(app)  # Разрешаем кросс-доменные запросы для виджета

# Инициализация моделей
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=True,
    api_key=QDRANT_API_KEY
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_context_from_qdrant(query, top_n=3):
    """Поиск релевантных чанков в Qdrant"""
    vec = embedding_model.encode(query).tolist()
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_n
    )
    context = "\n\n".join([hit.payload.get("text", "") for hit in results])
    return context

def amvera_gpt_query(user_question, context, token):
    """Запрос к GPT-5 Amvera с контекстом из Qdrant"""
    prompt = f"Вот выдержки из базы знаний:\n{context}\n\nОтветь на вопрос: \"{user_question}\" только используя эти фрагменты."
    
    headers = {
        "X-Auth-Token": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-5",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }
    
    response = requests.post(AMVERA_GPT_URL, json=payload, headers=headers, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.route("/", methods=["GET"])
def home():
    """Проверка работоспособности API"""
    return jsonify({"status": "ok", "message": "RAG Chatbot API is running"})

@app.route("/api/chat", methods=["POST"])
def chat():
    """Основной endpoint для обработки вопросов"""
    try:
        data = request.json
        user_question = data.get("question")
        
        if not user_question:
            return jsonify({"error": "Question is required"}), 400
        
        # Получаем контекст из Qdrant
        context = get_context_from_qdrant(user_question)
        
        # Генерируем ответ через GPT-5
        answer = amvera_gpt_query(user_question, context, AMVERA_GPT_TOKEN)
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
