FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Минимальные системные пакеты
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Python зависимости (кэшируем слой)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Предзагрузка той же модели, что в app.py (1024-мерные эмбеддинги)
RUN python - <<'PY'\nfrom sentence_transformers import SentenceTransformer\nSentenceTransformer('sberbank-ai/sbert_large_nlu_ru')\nPY

# Код приложения
COPY . .

# Безопасность: нерутовый пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Запуск (gunicorn.config.py уже в репозитории)
CMD ["gunicorn", "--config", "gunicorn.config.py", "app:app"]

