FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install gunicorn
RUN which gunicorn && gunicorn --version
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .

EXPOSE 8000

CMD gunicorn --bind 0.0.0.0:8000 --timeout 120 app:app
