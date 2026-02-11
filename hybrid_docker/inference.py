import os
import time
import random
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import scipy.sparse as sp
import joblib
from fastapi import FastAPI, HTTPException, Query, Request   
import uvicorn
from lightfm import LightFM
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response  
import json
import uuid
# =============================
# Prometheus metrics (GLOBAL)
# =============================

# HTTP-level metrics
HTTP_REQUESTS = Counter(
    "lightfm_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)

HTTP_REQUEST_LATENCY = Histogram(
    "lightfm_http_request_latency_seconds",
    "Latency for HTTP requests",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

# Model-specific metrics
INFERENCE_ERRORS = Counter(
    "lightfm_inference_errors_total",
    "Number of errors during recommendation",
    ["endpoint", "error_type"],
)

COLDSTART_USERS = Counter(
    "lightfm_coldstart_users_total",
    "Number of times cold-start user logic was used",
)

RETURNING_USERS = Counter(
    "lightfm_returning_users_total",
    "Number of times pre-existing user logic was used",
)

MODEL_LOAD_SECONDS = Gauge(
    "lightfm_model_load_seconds",
    "Time taken to load model + dataset + features at startup",
)

# -----------------------------
# Cold-start helper
# -----------------------------
def build_coldstart_user_row(dataset, tokens: List[str]) -> sp.csr_matrix:
    """
    Build a 1 x n_user_feature_cols CSR aligned to the training vocab.
    'tokens' are the same strings you used at training time
    (e.g., 'age:25-34', 'job:engineer', 'gender:M').
    """
    fmap = dataset._user_feature_mapping   # token -> col index (private API)
    cols = [fmap[t] for t in tokens if t in fmap]

    if not cols:
        # No tokens matched: return an all-zero row (model will fall back to biases)
        return sp.csr_matrix((1, len(fmap)))

    data = [1.0] * len(cols)
    return sp.csr_matrix(
        (data, ([0] * len(cols), cols)),
        shape=(1, len(fmap)),
    )


# -----------------------------
# Model state loading
# -----------------------------
def load_state(model_dir: str) -> Dict[str, Any]:
    """
    Load model and related artifacts once at startup.
    """
    model_path = os.path.join(model_dir, "lightfm_model.joblib")
    dataset_path = os.path.join(model_dir, "lightfm_dataset.joblib")
    user_feat_path = os.path.join(model_dir, "user_features.joblib")
    item_feat_path = os.path.join(model_dir, "item_features.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    start = time.time()
    model: LightFM = joblib.load(model_path)
    dataset = joblib.load(dataset_path)
    user_features = joblib.load(user_feat_path)
    item_features = joblib.load(item_feat_path)
    end = time.time()


    MODEL_LOAD_SECONDS.set(end - start)

    # Maps: internal index -> external item ID
    inv_item_map = {v: k for k, v in dataset._item_id_mapping.items()}

    print(f"[load_state] Loaded model and artifacts in {end - start:.3f} s")

    return {
        "model": model,
        "dataset": dataset,
        "user_features": user_features,
        "item_features": item_features,
        "inv_item_map": inv_item_map,
        "user_to_idx": dataset._user_id_mapping,
    }


# -----------------------------
# Core recommendation logic
# -----------------------------
def compute_recommendations(
    user_id: str,
    state: Dict[str, Any],
    top_k: int = 20,
    num_threads: int = 4,
) -> List[str]:
    """
    Automatically:
      - If user_id exists in dataset -> use trained user embedding
      - Else                         -> use cold-start tokens

    Returns a list of external item IDs.
    """
    model: LightFM = state["model"]
    dataset = state["dataset"]
    user_features = state["user_features"]
    item_features = state["item_features"]
    inv_item_map = state["inv_item_map"]
    user_to_idx = state["user_to_idx"]

    n_items = item_features.shape[0]
    item_ids = np.arange(n_items, dtype=np.int32)

    user_exists = user_id in user_to_idx

    if user_exists:
        # ---------- EXISTING USER PATH ----------
        u_idx = user_to_idx[user_id]  # internal LightFM index
        user_ids = np.full(n_items, u_idx, dtype=np.int32)

        scores = model.predict(
            user_ids=user_ids,
            item_ids=item_ids,
            user_features=user_features,   # full matrix from training
            item_features=item_features,
            num_threads=num_threads,
        )
    else:
        # ---------- COLD-START USER PATH ----------
        gender = ["M", "F"]
        jobs = [
            "other or not specified", "executive/managerial", "sales/marketing",
            "college/grad student", "doctor/health care", "academic/educator",
            "homemaker", "K-12 student", "self-employed", "scientist",
            "technician/engineer", "clerical/admin", "artist",
            "tradesman/craftsman", "retired", "unemployed", "programmer",
            "customer service", "writer", "lawyer", "farmer",
        ]
        age = ["25-34", "18-24", "35-44", "55-64", "45-54", "65+", "0-17"]

        tokens = [
            f"age:{random.choice(age)}",
            f"job:{random.choice(jobs)}",
            f"gender:{random.choice(gender)}",
        ]
        COLDSTART_USERS.inc()
        U_row = build_coldstart_user_row(dataset, tokens)
        user_ids = np.zeros(n_items, dtype=np.int32)  # all rows refer to row 0 of U_row

        scores = model.predict(
            user_ids=user_ids,
            item_ids=item_ids,
            user_features=U_row,           # 1×F cold-start vector
            item_features=item_features,
            num_threads=num_threads,
        )

    # Top-k indices -> external item IDs
    top_idx = scores.argsort()[::-1][:top_k]
    top_items = [inv_item_map[i] for i in top_idx]

    return top_items


# -----------------------------
# FastAPI app factory
# -----------------------------
def boot_api(state: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Shrekommender System hybrid (LightFM)")
    metadata = state.get("metadata", {})
    # ============ Prometheus middleware ============  # NEW
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        elapsed = time.time() - start_time

        endpoint = request.url.path

        HTTP_REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=response.status_code,
        ).inc()

        return response
    # ===============================================

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/recommend/{user_id}")
    def recommend_endpoint(
        user_id: str,
        top_k: int = Query(20, ge=1, le=100),
    ) -> Dict[str, Any]:
        try:
            recs = compute_recommendations(user_id, state, top_k=top_k)
        except Exception as e:
            INFERENCE_ERRORS.labels(
                endpoint="/recommend",
                error_type=type(e).__name__,
            ).inc()
            raise HTTPException(status_code=500, detail=str(e))
        request_id = str(uuid.uuid4())
        log_record = {
            "event": "recommendation",
            "timestamp": time.time(),
            "request_id": request_id,
            "user_id": user_id,
            "top_k": top_k,
            "model_id": metadata.get("model_id"),
            "model_version": metadata.get("model_version"),
            "train_code_commit": metadata.get("train_code_commit"),
            "pipeline_version": metadata.get("pipeline_version"),
            "data_provenance": metadata.get("data_provenance"),
        }
        # Structured log to stdout; hook this into whatever log pipeline you have
        print(json.dumps(log_record))







        return {
            "user_id": user_id,
            "top_k": top_k,
            "recommendations": recs,
        }
    
    @app.get("/metrics")
    def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return app
    


# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LightFM online inference API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("MODEL_DIR", "./"),
        help="Directory containing lightfm_model.joblib and friends.",
    )
    args = parser.parse_args()

    state = load_state(args.model_dir)
    app = boot_api(state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
