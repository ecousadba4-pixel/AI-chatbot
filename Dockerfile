FROM python:3.11-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    APP_DATA_DIR=/app/Data \
    EMBEDDING_MODEL_NAME=sberbank-ai/sbert_large_nlu_ru

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

# Базовые утилиты и прогрев модели (если веса уже на месте)
COPY embedding_loader.py ./embedding_loader.py
COPY tools ./tools
RUN python -m tools.preload_model

# Код приложения
COPY . .

# Безопасность: нерутовый пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Запуск (gunicorn.config.py уже есть)
CMD ["gunicorn", "--config", "gunicorn.config.py", "app:app"]
