# Choose whatever version you like
FROM python:3.10-slim

# System deps (optional, add as needed e.g. build-essential, libsndfile, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Work directory inside the container
WORKDIR /app

# Install Python deps first (cache-friendly)
RUN pip install --no-cache-dir \
    annotated-types==0.7.0 \
    asttokens  \
    certifi==2025.8.3 \
    charset-normalizer==3.4.3 \
    comm  \
    debugpy  \
    decorator  \
    exceptiongroup  \
    executing  \
    fastapi \
    uvicorn \
    idna==3.10 \
    importlib_metadata  \
    ipykernel  \
    ipython  \
    jedi  \
    joblib==1.5.2 \
    jupyter_client  \
    jupyter_core  \
    lightfm==1.17 \
    matplotlib-inline  \
    mypy_extensions==1.1.0 \
    nest_asyncio  \
    numpy==2.2.6 \
    packaging  \
    pandas==2.3.2 \
    pandera==0.26.1 \
    parso  \
    pexpect  \
    pickleshare  \
    platformdirs  \
    prompt_toolkit  \
    psutil  \
    ptyprocess  \
    pure_eval  \
    pydantic==2.12.3 \
    pydantic_core==2.41.4 \
    Pygments  \
    python-dateutil  \
    pytz==2025.2 \
    pyzmq  \
    requests==2.32.5 \
    scikit-learn==1.7.2 \
    scipy==1.15.3 \
    six  \
    stack_data  \
    threadpoolctl==3.6.0 \
    tornado  \
    tqdm==4.67.1\
    traitlets  \
    typeguard==4.4.4 \
    typing-inspect==0.9.0 \
    typing-inspection==0.4.2 \
    typing_extensions  \
    tzdata==2025.2 \
    urllib3==2.5.0 \
    wcwidth  \
    zipp \
    prometheus-client 


# Copy your source code
COPY . .

# Env var for model dir
ENV MODEL_DIR=/app/models

# Make sure the models directory exists
RUN mkdir -p ${MODEL_DIR}

ENV DATA_DIR=/app/data

RUN mkdir -p ${DATA_DIR}
# Expose API port (for inference)
EXPOSE 8080

# Default: run inference API
# You can change this depending on your framework
# Example if inference.py starts a FastAPI via uvicorn:
# CMD ["sh", "-c", "python train.py \
#   --users ${DATA_DIR}/user_data.csv \
#   --movies ${DATA_DIR}/movies_metadata.csv \
#   --ratings ${DATA_DIR}/ratings_snap.csv \
#   --epochs 5 \
#   --no_components 15 && inference.py"]

CMD ["sh", "-c", "python train.py \
    --users ${DATA_DIR}/user_data.csv \
    --movies ${DATA_DIR}/movies_metadata.csv \
    --ratings ${DATA_DIR}/ratings_snap.csv \
    --epochs 5 \
    --no_components 15 && \
    python inference.py --host 0.0.0.0 --port 8080 --model-dir ${MODEL_DIR}"]
#If inference.py uses Flask or FastAPI with uvicorn, just adapt the CMD accordingly, e.g.:
#CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]