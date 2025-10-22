from flask import Flask, request, jsonify
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Настройки Qdrant и GPT-5 из переменных окружения Amvera
QDRANT_HOST = os.getenv("QDRANT_HOST", "u4s-ai-chatbot-karinausadba.amvera.io")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 443))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hoteldocs")

AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL", "https://kong-proxy.yc.amvera.ru/api/v1/models/gpt-5")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "")

# Клиенты
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=True,
    api_key=QDRANT_API_KEY
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Получение контекста из Qdrant
def get_context_from_qdrant(query, top_n=3):
    vec = embedding_model.encode(query).tolist()
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_n
    )
    context = " ".join(hit.payload.get("text", "") for hit in results)
    return context

# Отправка запроса в GPT
def amvera_gpt_query(user_question, context, token):
    prompt = f"Контекст: {context}\n\nВопрос: {user_question}"
    headers = {
        "X-Auth-Token": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 512
    }
    response = requests.post(AMVERA_GPT_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "Извините, не удалось получить ответ.")
    else:
        return f"Ошибка GPT API: {response.text}"

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Некорректный запрос"}), 400

    context = get_context_from_qdrant(question)
    answer = amvera_gpt_query(question, context, AMVERA_GPT_TOKEN)
    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)


