FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------
# SYSTEM DEPENDENCIES
# Устанавливаем ТОЛЬКО необходимые пакеты
# ------------------------
RUN apt-get clean && apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ------------------------
# PYTHON DEPENDENCIES
# КРИТИЧНО: Копируем только requirements.txt ОТДЕЛЬНО
# Этот слой будет кэшироваться при изменении .py файлов
# ------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ------------------------
# DOWNLOAD MODEL CACHE (опционально)
# Предзагрузка модели sentence-transformers для ускорения старта
# ------------------------
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# ------------------------
# COPY PROJECT
# Только ПОСЛЕ установки зависимостей копируем код
# При изменении .py файлов пересоберётся только этот слой
# ------------------------
COPY . .

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "180", "--workers", "2", "--config", "gunicorn.config.py", "app:app"]
