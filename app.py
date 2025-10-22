from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app)  # Разрешаем внешние запросы с сайта

# Настройки Qdrant и Amvera (из переменных окружения)
QDRANT_HOST = os.getenv("QDRANT_HOST", "u4s-ai-chatbot-karinausadba.amvera.io")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 443))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hoteldocs")
AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL", "https://kong-proxy.yc.amvera.ru/api/v1/models/gpt-5")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "")

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=True,
    api_key=QDRANT_API_KEY
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_context_from_qdrant(query, top_n=3):
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
        print("Qdrant error:", str(e))
        return ""

def amvera_gpt_query(user_question, context, token):
    payload = {
        "model": "gpt-5",
        "messages": [
            {"role": "user", "text": f"{context}\n\n{user_question}"}
        ]
    }
    headers = {
        "X-Auth-Token": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(AMVERA_GPT_URL, headers=headers, json=payload, timeout=40)
        resp_json = response.json() if response.content else {}
        # ОСНОВНОЙ ФИКС: достаем текст ответа только из нужного поля OpenAI/Amvera
        answer = (
            (resp_json.get("choices", [{}])[0].get("message", {}).get("content")
                if resp_json.get("choices") else None)
            or resp_json.get("content")
            or resp_json.get("result")
            or (resp_json.get("choices", [{}])[0].get("text") if resp_json.get("choices") else None)
        )
        return answer or f"Ошибка GPT API: {response.text}"
    except Exception as e:
        print("Amvera error:", str(e))
        return "Ошибка ответа от сервера."

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Пожалуйста, задайте вопрос."}), 200
    print(f"Вопрос пользователя: {question}")
    context = get_context_from_qdrant(question)
    print(f"Контекст: {context[:120]}...") # для отладки
    answer = amvera_gpt_query(question, context, AMVERA_GPT_TOKEN)
    print(f"Ответ: {answer[:120]}...") # для отладки
    return jsonify({"answer": answer}), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 80)))




