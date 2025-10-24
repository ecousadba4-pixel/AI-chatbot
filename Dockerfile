# ---------- Базовый слой ----------
FROM python:3.11-slim

# Переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---------- Устанавливаем зависимости системы ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# ---------- Копируем requirements и проект ----------
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# ---------- Установка библиотек Python ----------
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------- Экспонируем порт ----------
EXPOSE 8000

# ---------- Команда запуска ----------
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "180", "app:app"]

