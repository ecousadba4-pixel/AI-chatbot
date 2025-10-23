from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os

# Добавляем импорт для диалога по расчёту стоимости
from price_dialog import handle_price_dialog

app = Flask(__name__)
CORS(app)

# --------------------- ENV ---------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "u4s-ai-chatbot-karinausadba.amvera.io")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 443))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hotel_docs")

AMVERA_GPT_URL = os.getenv("AMVERA_GPT_URL", "")
AMVERA_GPT_TOKEN = os.getenv("AMVERA_GPT_TOKEN", "")

# ------------------- Qdrant ---------------------
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=True,
    api_key=QDRANT_API_KEY
)

embedding_model = SentenceTransformer("sergeyzh/rubert-mini-frida")

# ------------------- Qdrant Context ---------------------
def get_context_from_qdrant(query, top_n=5):
    try:
        vec = embedding_model.encode(query).tolist()
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_n
        )
        context = " ".join(hit.payload.get("text", "") for hit in results)
        print(f"\n====== Контекст для GPT ======\n{context[:500]}\n==============================\n")
        return context or "Нет подходящих документов."
    except Exception as e:
        print("Ошибка Qdrant:", str(e))
        return "Ошибка при получении контекста."

# ------------------- GPT ---------------------
def amvera_gpt_query(user_question, context, token):
    payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "system",
                "text": (
                    "Ты чат-бот отеля. "
                    "Отвечай только на основе контекста. "
                    "Если данных нет — напиши, что информации нет. "
                    "Отвечай лаконично, до трёх предложений."
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

# ------------------- API ENDPOINTS ---------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.get_json(force=True).get("question", "").strip()
    user_id = request.get_json(force=True).get("session_id", "anon")

    if not user_input:
        return jsonify({"answer": "Пожалуйста, задайте вопрос."}), 200

    # 1. Проверка — диалог о стоимости проживания (через Shelter API)
    dialog_response = handle_price_dialog(user_id, user_input)
    if dialog_response:
        return jsonify(dialog_response), 200

    # 2. Общие ответы через Qdrant + GPT
    context = get_context_from_qdrant(user_input, top_n=5)
    answer = amvera_gpt_query(user_input, context, AMVERA_GPT_TOKEN)
    print(f"Ответ GPT: {answer[:300]}")
    return jsonify({"answer": answer, "mode": "info"}), 200

# ------------------- HEALTH CHECK ---------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ------------------- MAIN ---------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))





