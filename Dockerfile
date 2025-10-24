# ------------------------
# Dockerfile (optimized for RAG chatbot)
# ------------------------

# Этап 1 — базовый слой с Python
FROM python:3.11-slim as base

# Настройка окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

# Рабочая директория
WORKDIR /app

# Установка системных пакетов (для numpy и sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libglib2.0-0 \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Установка зависимостей Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем всё приложение
COPY . .

# ------------------------
# Этап 2 — продакшеновый слой
# ------------------------
FROM base as production

# Открываем порт
EXPOSE 8000

# Команда запуска Gunicorn
CMD ["gunicorn", "-c", "gunicorn.config.py", "app:app"]
