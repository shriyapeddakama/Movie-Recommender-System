FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAW_JSONL=/app/data/raw/watch.jsonl \
    ARTEFACTS_DIR=/app/artifacts \
    PORT=8080

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gfortran libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy scipy implicit fastapi uvicorn prometheus-client

ENV PATH="/opt/venv/bin:$PATH"

COPY als/als.py /app/als.py

RUN mkdir -p "${ARTEFACTS_DIR}" /app/data/raw

EXPOSE 8080

CMD ["sh", "-c", "python /app/als.py --raw-jsonl ${RAW_JSONL:?Set RAW_JSONL env or bake the file} --artefacts-dir ${ARTEFACTS_DIR} --host 0.0.0.0 --port ${PORT}"]
