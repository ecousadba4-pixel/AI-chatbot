FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Копируем весь код приложения
COPY . .

# Открываем порт 80 (Amvera ожидает порт 80)
EXPOSE 8000

# Запуск через gunicorn для продакшена
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "--timeout", "120", "app:app"]
