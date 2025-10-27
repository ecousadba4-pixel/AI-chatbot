# -*- coding: utf-8 -*-
"""
ingest_and_search_qdrant_ru.py
--------------------------------
Заливает JSON из ./processed в Qdrant (Amvera/локально) и позволяет сделать гибридный поиск.

Особенности:
- Эмбеддинги для ai-forever/sbert-base-lite-nlu-ru-v2 через sentence-transformers (mean pooling + L2).
- Кэш энкодера (модель грузится один раз).
- Автосборка QDRANT_URL из QDRANT_HOST/QDRANT_PORT/QDRANT_HTTPS, если QDRANT_URL не задан.
- Безопасное пересоздание коллекции (delete -> create), batch-upsert.
- Нормализованные эмбеддинги (COSINE).
- Гибридный поиск: семантика (Qdrant) + BM25-переранжировка по тексту документа (payload["text_bm25"]),
  с фильтрами must по category/source в Qdrant и финальным смешиванием скорингов.
- Удобный вывод: человекочитаемый и JSON (--json).

CLI:
    --ingest                 заливка данных из ./processed
    --recreate               пересоздать коллекцию перед заливкой
    --query "текст"          быстрый поиск
    --cat rooms|faq|...      фильтр категории (must)
    --source "Частые вопросы" фильтр источника (must)
    --limit 5                сколько вернуть после гибридного ранжирования
    --topk 50                сколько кандидатов взять из Qdrant на переранжировку BM25
    --alpha 0.6              вес семантики в финальном скоре (0..1)
    --json                   печатать JSON результат

Требования:
    pip install qdrant-client sentence-transformers torch python-dotenv numpy
"""

import os
import sys
import json
import argparse
import uuid
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from qdrant_client.http.exceptions import UnexpectedResponse

# Ensure local executions can import shared helpers irrespective of the entrypoint location.
_HERE = Path(__file__).resolve().parent
_HELPER_IMPORTED = False
for candidate in [_HERE, *_HERE.parents]:
    helper = candidate / "embedding_loader.py"
    if helper.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            from embedding_loader import resolve_embedding_model  # type: ignore
        except Exception:
            continue
        else:
            _HELPER_IMPORTED = True
            break

if not _HELPER_IMPORTED:
    # Fallback: provide a minimal inline resolver so the script remains usable
    # when embedding_loader.py is not bundled together with the script
    # (например, при ручном копировании ingest_and_search_qdrant_ru.py).
    from sentence_transformers import SentenceTransformer

    def resolve_embedding_model(*, model_name: str, allow_download: bool = True):
        """Минимальный загрузчик модели эмбеддингов."""

        if not allow_download:
            raise FileNotFoundError(
                "Загрузка модели из облака запрещена, альтернативы не предусмотрены."
            )

        return SentenceTransformer(model_name)



# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация путей и коллекции
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

# Модель эмбеддингов по умолчанию скачиваем из Hugging Face
DEFAULT_EMBEDDING_MODEL_NAME = "ai-forever/sbert-base-lite-nlu-ru-v2"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME
)

# ─────────────────────────────────────────────────────────────────────────────
# Загрузка .env и сборка QDRANT_URL
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")

def _as_bool_env(x: Optional[str], default=True) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


host = os.getenv("QDRANT_HOST")                     # напр.: u4s-ai-chatbot-karinausadba.amvera.io
port = os.getenv("QDRANT_PORT")                     # напр.: 443
https_flag = _as_bool_env(os.getenv("QDRANT_HTTPS"), True)

if host:
    scheme = "https" if https_flag else "http"
    if port and port.strip():
        default_url = f"{scheme}://{host}:{port}"
    else:
        default_url = f"{scheme}://{host}"
else:
    default_url = "http://localhost:6333"           # локальный дефолт

QDRANT_URL = os.getenv("QDRANT_URL", default_url)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")        # может быть пустым на локали
COLLECTION = os.getenv("QDRANT_COLLECTION") or os.getenv("COLLECTION_NAME", "hotel_ru")

# ─────────────────────────────────────────────────────────────────────────────
# Encoder: SentenceTransformer с единым загрузчиком
# ─────────────────────────────────────────────────────────────────────────────
_ENCODER_SINGLETON = None  # кэш энкодера




def get_encoder():
    global _ENCODER_SINGLETON
    if _ENCODER_SINGLETON is None:
        _ENCODER_SINGLETON = resolve_embedding_model(
            model_name=EMBEDDING_MODEL_NAME,
            allow_download=True,
        )
        dim = _ENCODER_SINGLETON.get_sentence_embedding_dimension()
        resolved = getattr(_ENCODER_SINGLETON, "_resolved_from", EMBEDDING_MODEL_NAME)
        print(f"[Encoder] Загружена модель {resolved} (dim={dim})")
    return _ENCODER_SINGLETON

# ─────────────────────────────────────────────────────────────────────────────
# Работа с Qdrant
# ─────────────────────────────────────────────────────────────────────────────
def qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return client

def check_qdrant_alive(client: QdrantClient):
    try:
        st = client.get_collections()
        print(f"[Qdrant] OK, доступно. Коллекций: {len(st.collections)}")
    except Exception as e:
        print(f"[Qdrant] Недоступно: {e}\nПроверь QDRANT_URL={QDRANT_URL}, API-ключ, сеть/файрвол и что сервис запущен.")
        raise

def recreate_collection_safe(client: QdrantClient, name: str, vector_size: int, distance=Distance.COSINE):
    """Удаляем коллекцию, если есть, затем создаём заново."""
    try:
        client.delete_collection(name)
        print(f"[Qdrant] Удалена коллекция: {name}")
    except UnexpectedResponse:
        print(f"[Qdrant] Коллекции {name} не было — ок")
    client.create_collection(collection_name=name, vectors_config=VectorParams(size=vector_size, distance=distance))
    print(f"[Qdrant] Создана коллекция: {name} (size={vector_size}, distance={distance})")

def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    """Создаём коллекцию, если её нет."""
    cols = client.get_collections().collections
    if not any(c.name == name for c in cols):
        client.create_collection(collection_name=name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
        print(f"[Qdrant] Создана коллекция: {name}")
    else:
        print(f"[Qdrant] Коллекция уже существует: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных из ./processed
# ─────────────────────────────────────────────────────────────────────────────
def load_processed() -> Dict[str, List[Dict[str, Any]]]:
    """
    Ожидаем файлы:
      structured_rooms.json
      structured_concept.json
      structured_contacts.json
      structured_hotel.json
      structured_loyalty.json
      structured_faq.json
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    names = [
        "structured_rooms.json",
        "structured_concept.json",
        "structured_contacts.json",
        "structured_hotel.json",
        "structured_loyalty.json",
        "structured_faq.json",
    ]
    for name in names:
        path = PROCESSED_DIR / name
        if not path.exists():
            print(f"[DATA] Пропускаю (нет файла): {path}")
            result[name] = []
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            result[name] = data
            print(f"[DATA] Загрузил {name}: {len(data)} записей")
    return result

def _join_text_blocks(text_blocks: Dict[str, Any]) -> str:
    parts = []
    for k, v in text_blocks.items():
        if not v:
            continue
        if isinstance(v, str):
            parts.append(f"{k}: {v}")
        else:
            try:
                parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            except Exception:
                pass
    return "\n".join(parts)

def make_item_text(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Возвращает (title, text_for_embedding) — и этот текст кладём в payload["text_bm25"]
    для BM25-переранжировки.
    """
    cat = item.get("category", "")
    title = item.get("title") or item.get("subcategory") or item.get("id") or "Без названия"

    if cat == "rooms":
        tb = item.get("text_blocks") or {}
        nums = item.get("numbers") or {}
        feat = item.get("features") or {}
        text = (
            f"{title}\n"
            f"{_join_text_blocks(tb)}\n"
            f"Числа: {json.dumps(nums, ensure_ascii=False)}\n"
            f"Особенности: {json.dumps(feat, ensure_ascii=False)}"
        )
        return title, text

    if cat in ("concept", "hotel", "loyalty"):
        body = item.get("text") or ""
        sub = item.get("subcategory") or item.get("tag") or ""
        text = f"{title}\n{sub}\n{body}"
        return title, text

    if cat == "contacts":
        fields = {k: v for k, v in item.items() if k not in ("id", "category")}
        text = f"{title}\n{json.dumps(fields, ensure_ascii=False)}"
        return title, text

    if cat == "faq":
        q = item.get("question", "")
        a = item.get("answer", "")
        tags = item.get("tags", [])
        text = f"Вопрос: {q}\nОтвет: {a}\nТеги: {', '.join(tags)}"
        return title or q[:64], text

    # fallback
    text = json.dumps(item, ensure_ascii=False)
    return title, text

def iter_all_items(dataset: Dict[str, List[Dict[str, Any]]]) -> Iterable[Dict[str, Any]]:
    for _, items in dataset.items():
        for it in items:
            yield it

# ─────────────────────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────────────────────
def ingest(recreate: bool = False):
    client = qdrant_client()
    check_qdrant_alive(client)

    encoder = get_encoder()  # кэшированный энкодер
    vector_size = encoder.get_sentence_embedding_dimension()

    if recreate:
        recreate_collection_safe(client, COLLECTION, vector_size)
    else:
        ensure_collection(client, COLLECTION, vector_size)

    data = load_processed()

    batch_texts: List[str] = []
    batch_payloads: List[Dict[str, Any]] = []
    batch_ids: List[str] = []

    BATCH = 128
    total = 0

    def flush_batch():
        nonlocal batch_texts, batch_payloads, batch_ids
        if not batch_texts:
            return
        vecs = encoder.encode(
            batch_texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        pts = [
            PointStruct(id=pid, vector=vec.astype(np.float32, copy=False), payload=pl)
            for pid, vec, pl in zip(batch_ids, vecs, batch_payloads)
        ]
        client.upsert(collection_name=COLLECTION, points=pts)
        print(f"[INGEST] upsert {len(pts)}")
        batch_texts, batch_payloads, batch_ids = [], [], []

    for item in iter_all_items(data):
        title, text = make_item_text(item)
        payload = {
            "category": item.get("category"),
            "title": title,
            "source": item.get("source"),
            "text_bm25": text,   # важно для гибридного поиска
            "raw": item,         # полный объект
        }
        pid = str(uuid.uuid4())
        batch_ids.append(pid)
        batch_texts.append(text)
        batch_payloads.append(payload)
        total += 1

        if len(batch_texts) >= BATCH:
            flush_batch()

    flush_batch()
    print(f"[DONE] Ингест завершён. Всего документов: {total}")

# ─────────────────────────────────────────────────────────────────────────────
# Токенизация и BM25
# ─────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]+", flags=re.U)

def _norm_token(s: str) -> str:
    s = s.lower().replace("ё", "е")
    return s

def _tokenize_ru(text: str) -> List[str]:
    return [_norm_token(m.group(0)) for m in _TOKEN_RE.finditer(text or "")]

def _bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    BM25 по списку документов docs (малый пул кандидатов).
    Возвращает список оценок по порядку docs.
    """
    q_tokens = _tokenize_ru(query)
    q_terms = list(dict.fromkeys(q_tokens))  # уникальные, сохраняем порядок

    # Подсчёты
    D = len(docs)
    doc_tokens = [_tokenize_ru(d) for d in docs]
    doc_len = [len(toks) for toks in doc_tokens]
    avgdl = (sum(doc_len) / D) if D > 0 else 1.0

    # df по пулу
    df = {t: 0 for t in q_terms}
    for toks in doc_tokens:
        st = set(toks)
        for t in q_terms:
            if t in st:
                df[t] += 1

    # idf
    idf = {}
    for t in q_terms:
        # добавим сглаживание, чтобы не было деления на 0 / отрицательности чрезмерной
        # классический BM25: idf = log( (D - df + 0.5) / (df + 0.5) + 1 )
        idf[t] = math.log((D - df[t] + 0.5) / (df[t] + 0.5) + 1.0)

    scores = []
    for i, toks in enumerate(doc_tokens):
        tf = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        s = 0.0
        denom = 1 - b + b * (doc_len[i] / max(avgdl, 1e-9))
        for t in q_terms:
            if tf.get(t, 0) == 0:
                continue
            num = tf[t] * (k1 + 1)
            den = tf[t] + k1 * denom
            s += idf[t] * (num / den)
        scores.append(s)
    return scores

def _minmax_norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmax, vmin):
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]

# ─────────────────────────────────────────────────────────────────────────────
# Поиск (query_points) + гибрид (семантика + BM25)
# ─────────────────────────────────────────────────────────────────────────────
def search(query: str,
           limit: int = 5,
           where_category: Optional[str] = None,
           where_source: Optional[str] = None,
           topk: int = 50,
           alpha: float = 0.6,
           as_json: bool = False):
    """
    alpha — вес семантического скора (0..1). 1.0 = только семантика, 0.0 = только BM25.
    topk  — сколько кандидатов забираем из Qdrant для переранжировки BM25.
    """
    client = qdrant_client()
    check_qdrant_alive(client)

    encoder = get_encoder()
    qv = encoder.encode(
        [query],
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32, copy=False)

    must_conditions = []
    if where_category:
        must_conditions.append(FieldCondition(key="category", match=MatchValue(value=where_category)))
    if where_source:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=where_source)))

    q_filter = Filter(must=must_conditions) if must_conditions else None

    # Универсальный вызов: в разных версиях клиента параметр называется query_filter / filter
    try:
        res = client.query_points(
            collection_name=COLLECTION,
            query=qv,
            limit=topk,
            with_payload=True,
            query_filter=q_filter,   # новое имя аргумента
            with_vectors=False,
        )
    except TypeError:
        res = client.query_points(
            collection_name=COLLECTION,
            query=qv,
            limit=topk,
            with_payload=True,
            filter=q_filter,         # альтернативное имя аргумента
            with_vectors=False,
        )

    # Нормализация формата результата к списку (payload, sem_score)
    def iter_points(result):
        seq = result.points if hasattr(result, "points") else result
        for item in seq:
            score = None
            point = item
            if isinstance(item, tuple):
                cand = item[0]
                if hasattr(cand, "payload"):
                    point = cand
                    score = getattr(cand, "score", None)
                    if score is None and len(item) > 1 and isinstance(item[1], (int, float)):
                        score = item[1]
                else:
                    point = cand
                    if len(item) > 1 and isinstance(item[1], (int, float)):
                        score = item[1]

            payload = {}
            if hasattr(point, "payload"):
                payload = point.payload or {}
                score = score if score is not None else getattr(point, "score", None)
            elif isinstance(point, dict):
                payload = point
            else:
                continue

            yield payload, float(score) if isinstance(score, (int, float)) else None

    raw = list(iter_points(res))
    if not raw:
        print("\n[SEARCH] Результаты:\n  (пусто)")
        return

    # Семантические скоры → min-max нормировка
    sem_scores = [s if isinstance(s, (int, float)) else 0.0 for (_, s) in raw]
    sem_norm = _minmax_norm(sem_scores)

    # BM25 по text_bm25
    docs = [pl.get("text_bm25", "") for (pl, _) in raw]
    bm25 = _bm25_scores(query, docs)
    bm25_norm = _minmax_norm(bm25)

    # Смешивание
    alpha = max(0.0, min(1.0, alpha))
    blended = [alpha * s + (1 - alpha) * b for s, b in zip(sem_norm, bm25_norm)]

    # Сортируем по blended убыванию
    idx = list(range(len(raw)))
    idx.sort(key=lambda i: blended[i], reverse=True)

    final = []
    for i in idx[:max(1, limit)]:
        pl, sem = raw[i]
        final.append({
            "score_sem": sem_scores[i],
            "score_sem_norm": sem_norm[i],
            "score_bm25": bm25[i],
            "score_bm25_norm": bm25_norm[i],
            "score_blended": blended[i],
            "payload": pl
        })

    # Вывод
    if as_json or os.getenv("SEARCH_JSON") == "1":
        print(json.dumps(final, ensure_ascii=False, indent=2))
        return

    print("\n[SEARCH] Результаты (гибрид):")
    for i, r in enumerate(final, 1):
        pl = r["payload"]
        cat = pl.get("category") or "-"
        title = pl.get("title") or "(без названия)"
        src = pl.get("source") or "-"
        b = r["score_blended"]
        s = r["score_sem_norm"]
        k = r["score_bm25_norm"]
        if cat == "faq":
            raw_pl = pl.get("raw") or {}
            q_txt = raw_pl.get("question", "") or "-"
            a_txt = raw_pl.get("answer", "") or "-"
            print(f"{i:2d}. blend={b:.4f} (sem={s:.3f}, bm25={k:.3f}) | {cat} | {title} | source={src}\n"
                  f"    Q: {q_txt}\n"
                  f"    A: {a_txt}\n")
        else:
            print(f"{i:2d}. blend={b:.4f} (sem={s:.3f}, bm25={k:.3f}) | {cat} | {title} | source={src}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest/Search Qdrant (RU hotel KB, hybrid)")
    parser.add_argument("--ingest", action="store_true", help="Залить данные из ./processed в Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию перед заливкой")
    parser.add_argument("--query", type=str, default=None, help="Поиск: текст запроса")
    parser.add_argument("--cat", type=str, default=None, help="Фильтр категории (rooms/faq/…)")
    parser.add_argument("--source", type=str, default=None, help="Фильтр источника (payload.source)")
    parser.add_argument("--limit", type=int, default=5, help="Сколько вернуть результатов")
    parser.add_argument("--topk", type=int, default=50, help="Сколько кандидатов взять из Qdrant")
    parser.add_argument("--alpha", type=float, default=0.6, help="Вес семантики в смешивании (0..1)")
    parser.add_argument("--json", action="store_true", help="Печатать результат поиска в JSON")
    args = parser.parse_args()

    if args.ingest:
        ingest(recreate=args.recreate)

    if args.query:
        search(
            query=args.query,
            limit=args.limit,
            where_category=args.cat,
            where_source=args.source,
            topk=args.topk,
            alpha=args.alpha,
            as_json=args.json
        )

if __name__ == "__main__":
    main()
