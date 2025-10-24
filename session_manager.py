import os
import uuid
import time
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv


# =================== Настройки ===================
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "amvera-karinausadba-run-u4s-ai-chatbot")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "chat_sessions"


# =================== Подключение Qdrant ===================
class QdrantConnection:
    def __init__(self):
        self.client = None
        self.reconnect()

    def reconnect(self, retries=10, delay=3):
        """Пытается подключиться к Qdrant, пока не получится."""
        for attempt in range(1, retries + 1):
            try:
                self.client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    api_key=QDRANT_API_KEY,
                    https=False  # ключевая настройка для Amvera!
                )
                self.client.get_collections()
                print(f"✅ Подключено к Qdrant ({QDRANT_HOST}:{QDRANT_PORT})")
                return True
            except Exception as e:
                print(f"⚠️ Qdrant недоступен (попытка {attempt}/{retries}): {e}")
                time.sleep(delay)
        print("❌ Не удалось подключиться к Qdrant после нескольких попыток — RAG временно неактивен.")
        return False

    def ensure_connection(self):
        """Проверяет соединение перед каждым запросом."""
        try:
            self.client.get_collections()
        except Exception:
            print("🔄 Потеря соединения c Qdrant, пробую восстановить...")
            self.reconnect()


qdrant_conn = QdrantConnection()
client = qdrant_conn.client


# =================== Инициализация коллекции ===================
def init_collection():
    try:
        qdrant_conn.ensure_connection()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            print(f"📦 Создана коллекция {COLLECTION_NAME}")
        else:
            print(f"ℹ️ Коллекция {COLLECTION_NAME} уже существует.")
    except Exception as e:
        print(f"⚠️ Ошибка при инициализации коллекции: {e}")


init_collection()


# =================== Работа с сессиями ===================
def get_or_create_session(session_id):
    """Находит или создаёт новую сессию пользователя."""
    try:
        qdrant_conn.ensure_connection()
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
            ),
            limit=1
        )
        if scroll_result and scroll_result[0]:
            return scroll_result[0][0].payload
    except Exception as e:
        print(f"⚠️ Ошибка при поиске сессии: {e}")
    return {"session_id": session_id, "messages": []}


def save_message(session_id, role, text):
    """Сохраняет сообщение (user / bot) в Qdrant."""
    try:
        qdrant_conn.ensure_connection()
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
                    id=int(uuid.uuid4().int & (1 << 63) - 1),
                    vector=[0.0] * 768,
                    payload=payload
                )
            ]
        )
    except Exception as e:
        print(f"⚠️ Ошибка при сохранении сообщения: {e}")


def get_recent_messages(session_id, limit=10):
    """Возвращает последние сообщения пользователя."""
    try:
        qdrant_conn.ensure_connection()
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
            ),
            limit=limit
        )
        results = [r.payload for r in scroll_result[0]]
        return sorted(results, key=lambda x: x.get("timestamp", ""))
    except Exception as e:
        print(f"⚠️ Ошибка при получении сообщений: {e}")
        return []


def clear_session(session_id):
    """Удаляет все сообщения конкретной сессии."""
    try:
        qdrant_conn.ensure_connection()
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
                )
            )
        )
        print(f"🧹 Сессия {session_id} очищена.")
    except Exception as e:
        print(f"⚠️ Ошибка при очистке сессии: {e}")


