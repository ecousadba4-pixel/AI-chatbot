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

# Timeout для долгих запросов (важно для GPT и интеграций с внешними сервисами)
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

def _resolve_port(raw_value: str | None) -> int:
    """Получить порт, учитывая требования хостинга."""

    if raw_value is None:
        return 8000

    try:
        value = int(raw_value)
    except ValueError:
        print(
            "[gunicorn.config] Некорректное значение PORT. Используется значение по умолчанию 8000.",
            file=sys.stderr,
        )
        return 8000

    if not (0 < value < 65536):
        print(
            "[gunicorn.config] PORT должен быть в диапазоне 1-65535. Используется значение по умолчанию 8000.",
            file=sys.stderr,
        )
        return 8000

    return value


# Bind
bind = f"0.0.0.0:{_resolve_port(os.getenv('PORT'))}"
