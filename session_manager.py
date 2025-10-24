import os
import uuid
import time
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Загружаем переменные окружения (.env — локально, на Amvera ENV передаются автоматически)
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "amvera-karinausadba-run-u4s-ai-chatbot")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "chat_sessions"

client = None

# Инициализация подключения с автоматическим восстановлением
def connect_qdrant(retries=5, delay=5):
    global client
    for attempt in range(1, retries + 1):
        try:
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                https=False,  # обязательно для Amvera
            )
            client.get_collections()  # тест запроса
            print(f"✅ Qdrant доступен ({QDRANT_HOST}:{QDRANT_PORT})")
            return
        except Exception as e:
            print(f"⚠️ Не удалось подключиться к Qdrant (попытка {attempt}/{retries}): {e}")
            time.sleep(delay)
    print("❌ Qdrant недоступен, работа будет продолжена в ограниченном режиме")

connect_qdrant()

# Проверка и создание коллекции
def init_collection():
    if not client:
        print("⚠️ Инициализация коллекции пропущена — клиент отсутствует")
        return
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            print(f"📦 Создана коллекция {COLLECTION_NAME}")
        else:
            print(f"ℹ️ Коллекция {COLLECTION_NAME} уже существует")
    except Exception as e:
        print(f"⚠️ Ошибка при инициализации коллекции: {e}")

init_collection()

# Создание или получение сессии пользователя
def get_or_create_session(session_id):
    """Возвращает существующую или создаёт новую сессию пользователя."""
    if not client:
        return {"session_id": session_id, "messages": []}
    try:
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
            ),
            limit=1,
        )
        if scroll_result and scroll_result[0]:
            return scroll_result[0][0].payload
    except Exception as e:
        print(f"⚠️ Ошибка при поиске сессии в Qdrant: {e}")
    return {"session_id": session_id, "messages": []}

# Сохранение сообщения
def save_message(session_id, role, text):
    """Сохраняет сообщение пользователя или ассистента в Qdrant."""
    if not client:
        return
    try:
        payload = {
            "session_id": session_id,
            "message_id": str(uuid.uuid4()),
            "role": role,
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        }
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().int & (1 << 63) - 1,
                    vector=[0.0] * 768,
                    payload=payload
                )
            ]
        )
    except Exception as e:
        print(f"⚠️ Ошибка при сохранении сообщения: {e}")

# Получение последних сообщений
def get_recent_messages(session_id, limit=10):
    """Извлекает последние сообщения из Qdrant."""
    if not client:
        return []
    try:
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
            ),
            limit=limit,
        )
        messages = [r.payload for r in scroll_result[0]]
        return sorted(messages, key=lambda x: x.get("timestamp", ""))
    except Exception as e:
        print(f"⚠️ Ошибка при получении сообщений: {e}")
        return []

# Очистка сессии
def clear_session(session_id):
    if not client:
        return
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
                )
            ),
        )
        print(f"🧹 Сессия {session_id} очищена")
    except Exception as e:
        print(f"⚠️ Ошибка при очистке сессии: {e}")

