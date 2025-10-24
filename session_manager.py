import os
import uuid
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Загружаем переменные окружения из .env (работает локально и на Amvera)
load_dotenv()

# Переменные окружения
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "chat_sessions"

# Инициализация клиента Qdrant
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    https=False  # очень важно в Amvera
)

# Проверка и создание коллекции
def init_collection():
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
        print(f"Создана коллекция {COLLECTION_NAME}")


# Получение или создание сессии пользователя
def get_or_create_session(user_token: str):
    now = datetime.utcnow()

    result, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_token", match=models.MatchValue(value=user_token))]
        ),
        limit=1
    )

    if result:
        return result[0].id

    session_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=session_id,
                vector=[0]*768,
                payload={
                    "user_token": user_token,
                    "messages": [],
                    "last_activity": now.isoformat(),
                    "ttl_expiration": (now + timedelta(days=60)).isoformat()
                }
            )
        ]
    )
    return session_id


# Сохранение сообщения (и пользователя, и ассистента)
def save_message(user_token: str, role: str, text: str):
    now = datetime.utcnow()
    session_id = get_or_create_session(user_token)

    existing, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_token", match=models.MatchValue(value=user_token))]
        ),
        limit=1
    )

    messages = existing[0].payload.get("messages", [])
    messages.append({"role": role, "text": text, "time": now.isoformat()})

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=session_id,
                vector=[0]*768,
                payload={
                    "user_token": user_token,
                    "messages": messages,
                    "last_activity": now.isoformat(),
                    "ttl_expiration": (now + timedelta(days=60)).isoformat()
                }
            )
        ]
    )


# Получение последних N сообщений для контекста
def get_recent_messages(user_token: str, limit: int = 5):
    existing, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_token", match=models.MatchValue(value=user_token))]
        ),
        limit=1
    )

    if not existing:
        return []
    messages = existing[0].payload.get("messages", [])
    return messages[-limit:]


# Удаление устаревших сессий (cron-задача)
def cleanup_expired_sessions():
    now = datetime.utcnow().isoformat()
    result, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000)

    for point in result:
        ttl = point.payload.get("ttl_expiration")
        if ttl and ttl < now:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsSelector(point_ids=[point.id])
            )


# Автоматическая инициализация при запуске модуля
init_collection()
