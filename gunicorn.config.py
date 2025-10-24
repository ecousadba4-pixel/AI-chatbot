import os
import multiprocessing

# Количество worker-процессов
workers = int(os.getenv("GUNICORN_WORKERS", "2"))

# Тип worker'ов
worker_class = "sync"

# Максимальное количество одновременных запросов на worker
threads = 2

# Timeout для долгих запросов (важно для GPT и Qdrant)
timeout = 180

# Graceful timeout при перезапуске
graceful_timeout = 30

# Keep-alive соединений
keepalive = 5

# Логирование
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload приложения (ускоряет запуск workers)
preload_app = True

# Bind
bind = "0.0.0.0:8000"
