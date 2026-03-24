FROM python:3.11-slim

# システム依存が必要なら適宜追加（例：git, build-essential など）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先に依存だけをコピー→インストール（レイヤーキャッシュ最適化）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 残りのソース
COPY . /app

EXPOSE 8501

# compose 側で command を上書きするため、ここはダミーでもOK
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]