FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------
# SYSTEM DEPENDENCIES (Debian 13 Trixie)
# ------------------------
RUN apt-get clean && apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libgl1 \
    libelf1t64 \
    liberror-perl \
    libexpat1 \
    libgbm1 \
    libgcc-14-dev \
    libgdbm-compat4t64 \
    libstdc++6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ------------------------
# PYTHON DEPENDENCIES
# ------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# ------------------------
# COPY PROJECT
# ------------------------
COPY . .

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "180", "app:app"]
