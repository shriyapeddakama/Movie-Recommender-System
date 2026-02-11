import os
import glob
import json
import pandas as pd
import requests


def fetch_movie_data(movie_id):
    url = f"http://128.2.220.241:8080/movie/{movie_id}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        return data
    else:
        print(f"Failed to fetch {movie_id}: {r.status_code}")


def fetch_user_data(user_id):
    url = f"http://128.2.220.241:8080/user/{user_id}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        return data
    else:
        print(f"Failed to fetch {user_id}: {r.status_code}")


USER_COLUMNS = ["user_id", "age", "occupation", "gender"]
MOVIE_COLUMNS = [
    "id",
    "tmdb_id",
    "imdb_id",
    "title",
    "original_title",
    "adult",
    "belongs_to_collection",
    "budget",
    "genres",
    "homepage",
    "original_language",
    "overview",
    "popularity",
    "poster_path",
    "production_companies",
    "production_countries",
    "release_date",
    "revenue",
    "runtime",
    "spoken_languages",
    "status",
    "vote_average",
    "vote_count",
]

RATE_COLUMNS = ["timestamp", "user_id", "event_type", "movie_id", "rating"]

VALID_EVENT_TYPES = {"watch", "rate"}


def ensure_csv(path, columns):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def ensure_data_files(users_csv, movies_csv):
    ensure_csv(users_csv, USER_COLUMNS)
    ensure_csv(movies_csv, MOVIE_COLUMNS)


def load_all_jsonl_from_raw(raw_data_dir):
    events = []

    for file in glob.glob(os.path.join(raw_data_dir, "*.jsonl")):
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[SKIP] Bad JSON in {file}: {line}")

    return events


def parse_event(raw_event):
    if isinstance(raw_event, str):
        try:
            data = json.loads(raw_event)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw_event, dict):
        data = raw_event
    else:
        return None

    event_type = data.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        return None

    user_id = data.get("user_id")
    movie_id = data.get("movie_id")
    if user_id is None or movie_id is None:
        return None

    timestamp = data.get("timestamp")

    event_record = {"user_id": user_id, "movie_id": movie_id, "timestamp": timestamp, "event_type": event_type}
    if event_type == "rate":
        event_record["rating"] = data.get("rating")
    return event_record


def load_existing_ids(csv_path, column_name):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    column_df = pd.read_csv(csv_path, usecols=[column_name])
    return set(column_df[column_name].astype(str).values)


def append_new_users(user_ids, users_csv):
    if not user_ids:
        return 0

    rows = []
    for uid in user_ids:
        if len(rows) % 100 == 0:
            print(f"[BATCH] Fetching metadata for user {uid} (#{len(rows) + 1})")
        metadata = fetch_user_data(uid) or {}
        rows.append(
            {
                "user_id": uid,
                "age": metadata.get("age", ""),
                "occupation": metadata.get("occupation", ""),
                "gender": metadata.get("gender", ""),
            }
        )

    users_df = pd.DataFrame(rows, columns=USER_COLUMNS)
    file_exists = os.path.exists(users_csv)
    needs_header = not file_exists or os.path.getsize(users_csv) == 0
    users_df.to_csv(users_csv, mode="a", header=needs_header, index=False)
    return len(rows)


def append_new_movie(movie_id, movies_csv):
    print(f"[BATCH] Fetching metadata for movie {movie_id}")
    metadata = fetch_movie_data(movie_id)
    if not metadata:
        metadata = {"title": str(movie_id).replace("+", " ").title(), "genres": "Unknown"}

    new_movie = {
        "id": movie_id,
        "tmdb_id": metadata.get("tmdb_id", ""),
        "imdb_id": metadata.get("imdb_id", ""),
        "title": metadata.get("title", ""),
        "original_title": metadata.get("original_title", ""),
        "adult": metadata.get("adult", False),
        "belongs_to_collection": metadata.get("belongs_to_collection", ""),
        "budget": metadata.get("budget", 0),
        "genres": metadata.get("genres", ""),
        "homepage": metadata.get("homepage", ""),
        "original_language": metadata.get("original_language", ""),
        "overview": metadata.get("overview", ""),
        "popularity": metadata.get("popularity", 0.0),
        "poster_path": metadata.get("poster_path", ""),
        "production_companies": metadata.get("production_companies", ""),
        "production_countries": metadata.get("production_countries", ""),
        "release_date": metadata.get("release_date", ""),
        "revenue": metadata.get("revenue", 0),
        "runtime": metadata.get("runtime", 0),
        "spoken_languages": metadata.get("spoken_languages", ""),
        "status": metadata.get("status", ""),
        "vote_average": metadata.get("vote_average", 0.0),
        "vote_count": metadata.get("vote_count", 0),
    }

    new_movie_df = pd.DataFrame([new_movie], columns=MOVIE_COLUMNS)
    file_exists = os.path.exists(movies_csv)
    needs_header = not file_exists or os.path.getsize(movies_csv) == 0
    new_movie_df.to_csv(movies_csv, mode="a", header=needs_header, index=False)
    return True


def save_rate_events(rate_events, rate_events_csv):
    """Save all rate events to CSV file (overwrites existing file)."""
    if not rate_events:
        return 0
    ensure_parent_dir(rate_events_csv)
    rate_df = pd.DataFrame(rate_events, columns=RATE_COLUMNS)
    rate_df.to_csv(rate_events_csv, index=False)
    return len(rate_events)


def preprocess_events(events, users_csv, movies_csv, rate_events_csv):
    ensure_data_files(users_csv, movies_csv)

    existing_user_ids = load_existing_ids(users_csv, "user_id")
    existing_movie_ids = load_existing_ids(movies_csv, "id")

    new_user_ids = []
    candidate_movies = {}
    rate_events = []
    skipped_events = 0
    processed_count = 0

    for raw_event in events:
        parsed = parse_event(raw_event)
        if not parsed:
            skipped_events += 1
            continue

        processed_count += 1
        if processed_count % 1000 == 0:
            print(
                f"[BATCH] Processed {processed_count} events "
                f"(candidates: {len(new_user_ids)} new users, {len(candidate_movies)} movies)"
            )

        if parsed["event_type"] == "rate" and parsed.get("rating") is not None and parsed.get("timestamp"):
            rate_events.append(
                {
                    "timestamp": parsed["timestamp"],
                    "user_id": parsed["user_id"],
                    "event_type": parsed["event_type"],
                    "movie_id": parsed["movie_id"],
                    "rating": parsed["rating"],
                }
            )

        user_value = parsed["user_id"]
        user_key = str(user_value)
        if user_key not in existing_user_ids:
            existing_user_ids.add(user_key)
            new_user_ids.append(user_value)

        movie_value = parsed["movie_id"]
        movie_key = str(movie_value)
        if movie_key not in candidate_movies:
            candidate_movies[movie_key] = movie_value

    added_users = append_new_users(new_user_ids, users_csv)

    added_movies = 0
    for movie_key, movie_value in candidate_movies.items():
        if movie_key in existing_movie_ids:
            continue
        if append_new_movie(movie_value, movies_csv):
            existing_movie_ids.add(movie_key)
            added_movies += 1

    saved_rates = save_rate_events(rate_events, rate_events_csv)

    print(f"[BATCH] Added {added_users} new users")
    print(f"[BATCH] Added {added_movies} new movies")
    print(f"[BATCH] Saved {saved_rates} rate events")
    if skipped_events:
        print(f"[BATCH] Skipped {skipped_events} unsupported or invalid events")


def run_batch(raw_dir):
    # Auto-derive processed directory from raw directory
    base_dir = os.path.dirname(raw_dir) or "."
    processed_dir = os.path.join(base_dir, "processed")
    users_csv = os.path.join(processed_dir, "users.csv")
    movies_csv = os.path.join(processed_dir, "movies.csv")
    rate_events_csv = os.path.join(processed_dir, "rate_events.csv")

    print("\n=== STEP 2: Load raw JSONL ===")
    events = load_all_jsonl_from_raw(raw_dir)
    print(f"Loaded {len(events)} total events")

    print("\n=== STEP 4: Preprocess events ===")
    preprocess_events(events, users_csv, movies_csv, rate_events_csv)

    print("\n=== DONE: Batch pipeline completed ===\n")


if __name__ == "__main__":
    import argparse

    def _default_env(name, default):
        """Get environment variable or return default."""
        return os.environ.get(name, default)

    parser = argparse.ArgumentParser(description="Batch preprocess raw event data")
    parser.add_argument(
        "--raw-dir",
        default=_default_env("RAW_DATA_DIR", "data/raw"),
        help="Directory containing raw *.jsonl files (default env RAW_DATA_DIR or %(default)s)",
    )

    args = parser.parse_args()
    run_batch(args.raw_dir)
