# ------------------------
# Base image
# ------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------
# Системные зависимости
# ------------------------
RUN apt-get clean && apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libstdc++6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ------------------------
# Python deps
# ------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ------------------------
# Copy project
# ------------------------
COPY . .

EXPOSE 8000

# ------------------------
# Start app
# ------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "180", "app:app"]
