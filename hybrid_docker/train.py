from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
from utils import process_data, train_model
from lightfm.cross_validation import random_train_test_split
import joblib
import argparse
import time
from prometheus_client import Gauge, start_http_server
import os
import hashlib
import subprocess
import datetime
import json
parser = argparse.ArgumentParser(description ='Process some integers.')
parser.add_argument('--users',
                    type = str, default='/home/team11/project_data/data/processed/user_data.csv',
                    help ='Path to csv containing users data.')
parser.add_argument('--movies',
                    type = str, default='/home/team11/project_data/data/processed/movies_metadata.csv',
                    help ='Path to csv containing movies data.')
parser.add_argument('--ratings',
                    type = str, default='/home/team11/project_data/data/processed/ratings_snap.csv',
                    help ='Path to csv containing ratings data.')
parser.add_argument('--epochs',
                    type = int, default=1,
                    help ='number of training epochs.')
parser.add_argument('--no_components',
                    type = int, default=15,
                    help ='number of components per feature.')
parser.add_argument('--learning_schedule',
                    type = str, default='adagrad',
                    help ='learning_schedule type to train the model [adagrad, adadelta].')
parser.add_argument('--loss',
                    type = str, default='warp',
                    help ='loss type to train the model [warp, logistic, bpr, warp-kos].')
parser.add_argument('--random_state',
                    type = int, default=42,
                    help ='seed for reproducing exp.')
parser.add_argument('--frac',
                    type = float, default=0.1,
                    help ='Percentage of training data.')

args = parser.parse_args()


TRAIN_FULL_SECONDS = Gauge(
    "lightfm_train_full_pipeline_seconds",
    "Total time for full LightFM training pipeline run",
)
TRAIN_DATA_SECONDS = Gauge(
    "lightfm_train_data_pipeline_seconds",
    "Time for dataset building (Dataset, interactions, split)",
)
TRAIN_PROCESS_SECONDS = Gauge(
    "lightfm_train_processing_seconds",
    "Time for initial processing (process_data)",
)


# ------------- Provenance helpers -------------
def compute_file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def get_git_commit() -> str:
    """Best-effort: try to get current git commit, otherwise env or 'unknown'."""
    # Prefer explicit env if you set it in Docker build/CI
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "unknown"

def build_model_metadata(run_dir: str, args, df_user, df_movie, df_rating) -> dict:
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    data_info = {}
    for name, path, df in [
        ("users", args.users, df_user),
        ("movies", args.movies, df_movie),
        ("ratings", args.ratings, df_rating),
    ]:
        info = {
            "path": os.path.abspath(path),
            "n_rows": int(len(df)),
        }
        try:
            info["sha256"] = compute_file_sha256(path)
        except Exception as e:
            info["sha256"] = f"error: {e}"
        data_info[name] = info

    metadata = {
        "model_id": "lightfm-hybrid",
        "model_version": timestamp,  # timestamp-based version
        "created_at_utc": timestamp,
        "train_code_commit": get_git_commit(),
        "pipeline_version": os.environ.get("PIPELINE_VERSION", "unknown"),
        "run_dir": os.path.abspath(run_dir),
        "train_args": {
            "epochs": args.epochs,
            "no_components": args.no_components,
            "learning_schedule": args.learning_schedule,
            "loss": args.loss,
            "random_state": args.random_state,
            "frac": args.frac,
        },
        "data_provenance": data_info,
    }
    return metadata

def main(args):
    start_http_server(8080)
    start_all= time.time()
    df_user= pd.read_csv(args.users)
    df_movie= pd.read_csv(args.movies)
    df_rating= pd.read_csv(args.ratings)

    #split train/test based on user
    
    print(f" Users: {len(df_user)}, Ratings: {len(df_rating)}, Movies: {len(df_movie)}")
    start_process= time.time()
    #process
    all_user_ids, all_item_ids, all_user_feats, all_item_feats, user_feats_iter, item_feats_iter, triples = process_data(df_user, df_movie, df_rating)
    end_process= time.time()
    #create datasets objects
    start_data= time.time()
    dataset= Dataset()
    dataset.fit(users=all_user_ids,
                items=all_item_ids,
                user_features= all_user_feats,
                item_features= all_item_feats)
    user_features = dataset.build_user_features(user_feats_iter, normalize=False)
    item_features = dataset.build_item_features(item_feats_iter, normalize=False)
    interactions, weights = dataset.build_interactions(triples)

    interactions_train, interactions_test = random_train_test_split(
    interactions, test_percentage=args.frac, random_state=args.random_state
    )
    end_data= time.time()


    #model
    model = LightFM(no_components=args.no_components, loss=args.loss, random_state=args.random_state, learning_schedule=args.learning_schedule, item_alpha=1e-6)
    
    train_model(model, interactions_train, interactions_test, [weights, user_features, item_features], args.epochs)
    end_all= time.time()
    end_all = time.time()

    full_time = end_all - start_all
    data_time = end_data - start_data
    process_time = end_process - start_process

    TRAIN_FULL_SECONDS.set(full_time)
    TRAIN_DATA_SECONDS.set(data_time)
    TRAIN_PROCESS_SECONDS.set(process_time)
    print(f"full pipline time: {start_all}s\tdataset pipline time: {data_time}s\tdataset processing: {process_time}s")
    
    model_root = os.environ.get("MODEL_DIR", "/app/models")
    os.makedirs(model_root, exist_ok=True)

    # Use UTC timestamp as version
    version_ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(model_root, f"lightfm_{version_ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    
    #save model
    joblib.dump(model, "/app/models/lightfm_model.joblib")
    joblib.dump(dataset, "/app/models/lightfm_dataset.joblib")
    joblib.dump(user_features, "/app/models/user_features.joblib")
    joblib.dump(item_features, "/app/models/item_features.joblib")

    metadata = build_model_metadata(run_dir, args, df_user, df_movie, df_rating)
    with open(os.path.join(run_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] Saved model and metadata to: {run_dir}")



if __name__ == "__main__":
    main(args)