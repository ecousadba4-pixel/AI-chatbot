# ------------------------
# Gunicorn configuration for Flask RAG Chatbot (Amvera)
# ------------------------

import multiprocessing

# Количество процессов равно числу CPU ядер (Amvera auto-scales)
workers = multiprocessing.cpu_count() * 2 + 1
threads = 4

# Хост и порт
bind = "0.0.0.0:8000"

# Тип воркеров — синхронный (Flask-friendly)
worker_class = "sync"

# Таймауты (GPT и Qdrant-запросы могут быть долгими)
timeout = 120
graceful_timeout = 30

# Очистка памяти после завершения запроса
preload_app = True

# Лимит запросов к одному воркеру (во избежание деградации от накопления модели)
max_requests = 1000
max_requests_jitter = 100

# Логирование
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Метка процесса, удобно для мониторинга
proc_name = "usadba-chatbot-rag"
