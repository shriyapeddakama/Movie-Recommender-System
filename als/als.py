"""Quick script: ingest JSONL, build ALS artefacts, train, and serve."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import implicit
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Response
from prometheus_client import Counter, Gauge, generate_latest
from scipy.sparse import csr_matrix, save_npz


MODEL_ENV_VAR = "ALS_MODEL_ENV"
DEFAULT_MODEL_ENV = "als@v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal preprocess -> train -> serve pipeline.")
    parser.add_argument("--raw-jsonl", type=Path, required=True, help="Raw watch JSONL file path.")
    parser.add_argument("--artefacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional path to the metadata.json produced by the ingestion stage.",
    )
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--model-name", default="als", help="Friendly name for the trained model.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def _resolve_metadata_path(raw_jsonl: Path, explicit_metadata: Path | None) -> Path | None:
    if explicit_metadata:
        return explicit_metadata
    candidate = raw_jsonl.parent / "metadata.json"
    return candidate if candidate.exists() else None


def load_ingestion_metadata(raw_jsonl: Path, explicit_metadata: Path | None) -> dict[str, object]:
    """Read ingestion metadata so we can expose data lineage via /model-info."""

    metadata_path = _resolve_metadata_path(raw_jsonl, explicit_metadata)
    if not metadata_path:
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[metadata] Unable to read {metadata_path}: {exc}")
        return {}

    metadata["_source_path"] = str(metadata_path)
    return metadata


def summarize_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if not metadata:
        return {}

    summary = {
        "ingested_at": metadata.get("run_timestamp"),
        "total_events": metadata.get("total_events"),
        "watch_events": metadata.get("watch"),
        "rate_events": metadata.get("rate"),
        "recommendation_events": metadata.get("recommendation"),
        "metadata_path": metadata.get("_source_path"),
    }
    files_created = metadata.get("files_created")
    if isinstance(files_created, list):
        summary["files_created"] = files_created
    return {k: v for k, v in summary.items() if v is not None}


def load_df(raw_jsonl: Path) -> pd.DataFrame:
    rows = []
    with raw_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Input file is empty.")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="ISO8601")
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("All rows were dropped because 'timestamp' could not be parsed.")
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] Dropped {dropped} rows with invalid timestamps.")
    return df.sort_values("timestamp").reset_index(drop=True)


def build_matrix(df_train: pd.DataFrame) -> tuple[csr_matrix, dict, dict, dict]:
    df_train = df_train.copy()
    df_train["user_id"] = df_train["user_id"].astype(str)
    df_train["movie_id"] = df_train["movie_id"].astype(str)

    users = sorted(df_train["user_id"].unique())
    movies = sorted(df_train["movie_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    movie_to_idx = {m: i for i, m in enumerate(movies)}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    grouped = df_train.groupby(["user_id", "movie_id"]).size().reset_index(name="count")
    rows = grouped["user_id"].map(user_to_idx).to_numpy()
    cols = grouped["movie_id"].map(movie_to_idx).to_numpy()
    data = np.ones(len(grouped), dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(movies)), dtype=np.float32)
    return matrix, user_to_idx, movie_to_idx, idx_to_movie


def write_segments(df_subset: pd.DataFrame, target: Path) -> None:
    with target.open("w", encoding="utf-8") as f:
        for _, row in df_subset.iterrows():
            payload = {
                "timestamp": row["timestamp"].isoformat(),
                "user_id": row["user_id"],
                "event_type": row.get("event_type"),
                "movie_id": row["movie_id"],
                "minute": row.get("minute"),
            }
            f.write(json.dumps(payload) + "\n")


def preprocess(raw_path: Path, artefacts_dir: Path):
    df = load_df(raw_path)
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    segments_path = artefacts_dir / "watch_segments.jsonl"
    write_segments(df, segments_path)

    matrix, user_to_idx, movie_to_idx, idx_to_movie = build_matrix(df)
    save_npz(artefacts_dir / "train_matrix.npz", matrix)

    with (artefacts_dir / "user_mappings.pkl").open("wb") as f:
        pickle.dump({"user_to_idx": user_to_idx, "idx_to_user": {idx: user for user, idx in user_to_idx.items()}}, f)

    with (artefacts_dir / "movie_mappings.pkl").open("wb") as f:
        pickle.dump({"movie_to_idx": movie_to_idx, "idx_to_movie": idx_to_movie}, f)

    stats = {
        "interactions": len(df),
        "users": len(user_to_idx),
        "movies": len(movie_to_idx),
    }
    with (artefacts_dir / "dataset_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    return matrix, user_to_idx, movie_to_idx, idx_to_movie, stats


def train(matrix: csr_matrix, factors: int, iterations: int, regularization: float, alpha: float):
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        alpha=alpha,
        random_state=42,
    )
    model.fit(matrix * alpha)
    movie_popularity = np.asarray(matrix.sum(axis=0)).ravel()
    popular_indices = np.argsort(-movie_popularity)
    return model, popular_indices


REQUESTS_TOTAL = Counter("als_recommend_requests_total", "Total recommendation requests processed")
COLD_START_TOTAL = Counter("als_recommend_cold_start_total", "Requests served via popular fallback")
TRAINED_USERS_GAUGE = Gauge("als_trained_users", "Number of users seen during training")
TRAINED_MOVIES_GAUGE = Gauge("als_trained_movies", "Number of movies seen during training")


def boot_api(state: dict[str, object]) -> FastAPI:
    TRAINED_USERS_GAUGE.set(len(state["user_to_idx"]))
    TRAINED_MOVIES_GAUGE.set(len(state["movie_to_idx"]))

    app = FastAPI(title="Shrekommender System (pipeline)")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/recommend/{user_id}")
    def recommend(user_id: str):
        top_k = 20
        REQUESTS_TOTAL.inc()
        if user_id not in state["user_to_idx"]:
            movies = [state["idx_to_movie"][idx] for idx in state["popular_indices"][:top_k]]
            COLD_START_TOTAL.inc()
            return ",".join(map(str, movies))

        user_idx = state["user_to_idx"][user_id]
        recommendations = state["model"].recommend(
            user_idx,
            user_items=state["train_matrix"],
            N=top_k,
            filter_already_liked_items=True,
        )
        movie_ids = [state["idx_to_movie"][idx] for idx, _ in recommendations][:top_k]
        return ",".join(map(str, movie_ids))

    @app.get("/metrics")
    def metrics() -> Response:
        payload = generate_latest()
        return Response(content=payload, media_type="text/plain; version=0.0.4")

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        return state["model_info"]

    return app


def main() -> None:
    args = parse_args()
    raw_path = args.raw_jsonl.resolve()
    artefacts_dir = args.artefacts_dir.resolve()
    metadata_override = args.metadata_json.resolve() if args.metadata_json else None
    metadata = load_ingestion_metadata(raw_path, metadata_override)
    data_info = summarize_metadata(metadata)

    matrix, user_to_idx, movie_to_idx, idx_to_movie, dataset_stats = preprocess(raw_path, artefacts_dir)
    model, popular_indices = train(matrix, args.factors, args.iterations, args.regularization, args.alpha)

    with (artefacts_dir / "als_model.pkl").open("wb") as fh:
        pickle.dump({"model_state": model.__getstate__()}, fh)

    state = {
        "model": model,
        "train_matrix": matrix,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
        "popular_indices": popular_indices,
        "model_info": {
            "model_name": args.model_name,
            "environment": os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL_ENV),
            "data_source": data_info,
            "dataset_stats": dataset_stats,
        },
    }

    app = boot_api(state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
