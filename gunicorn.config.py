import multiprocessing
import os
import sys


def _default_worker_count() -> int:
    """Подобрать безопасное значение workers по умолчанию."""

    cpu_count = multiprocessing.cpu_count() or 1
    # Для небольших инстансов одного процесса достаточно, иначе используем половину ядер
    return max(1, min(cpu_count, cpu_count // 2 or 1))


def _resolve_workers(raw_value: str | None) -> int:
    """Преобразовать значение переменной окружения в целое."""

    if raw_value is None:
        return _default_worker_count()

    try:
        value = int(raw_value)
    except ValueError:
        print(
            "[gunicorn.config] Некорректное значение GUNICORN_WORKERS. Используется значение по умолчанию.",
            file=sys.stderr,
        )
        return _default_worker_count()

    if value < 1:
        print(
            "[gunicorn.config] GUNICORN_WORKERS должно быть положительным. Используется значение по умолчанию.",
            file=sys.stderr,
        )
        return _default_worker_count()

    return value


# Количество worker-процессов
workers = _resolve_workers(os.getenv("GUNICORN_WORKERS"))

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
