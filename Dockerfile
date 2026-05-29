FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 依存だけ先に入れてレイヤキャッシュを効かせる
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# “No secrets files found” の赤帯を消すための空ファイル
RUN mkdir -p /app/.streamlit && printf "" > /app/.streamlit/secrets.toml
RUN apt-get update && apt-get install -y wget gnupg && rm -rf /var/lib/apt/lists/*
RUN playwright install chromium
RUN playwright install-deps chromium

# アプリ本体
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]