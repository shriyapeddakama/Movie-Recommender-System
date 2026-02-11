FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SHREK_KAFKA_BOOTSTRAP="localhost:9092" \
    SHREK_KAFKA_TOPIC="movielog11" \
    SHREK_KAFKA_GROUP="" \
    SHREK_MAX_EVENTS="5000" \
    RAW_DATA_DIR="/app/data/raw"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir kafka-python pandas requests

ENV PATH="/opt/venv/bin:$PATH"

COPY data_pipeline/simple_ingestion.py /app/simple_ingestion.py
COPY data_pipeline/batch_proprocess.py /app/batch_proprocess.py

RUN mkdir -p /app/data/raw /app/data/processed

CMD ["bash", "-c", "python /app/simple_ingestion.py && python /app/batch_proprocess.py"]
