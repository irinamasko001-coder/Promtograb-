FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .

RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /data \
    && chown -R bot:bot /app /data
USER bot

CMD ["sh", "-c", "uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080}"]

