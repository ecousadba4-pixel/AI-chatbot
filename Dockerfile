FROM python:3.11-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

ARG EMBEDDING_MODEL_NAME=d0rj/e5-base-en-ru
ENV EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}

WORKDIR /app

# Общая директория кэша для моделей, доступная на этапе билда и рантайма
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence_transformers

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

# Код приложения и утилиты
COPY . .

# Прогрев модели: загрузим веса из Hugging Face на этапе сборки контейнера
RUN python -m tools.preload_model

# Безопасность: нерутовый пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Запуск (gunicorn.config.py уже есть)
CMD ["gunicorn", "--config", "gunicorn.config.py", "app:app"]
