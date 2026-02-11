import os
import json
import pandas as pd
from datetime import datetime
from shrekommender_system.data.schema import parse_event, WatchEvent, RateEvent
from shrekommender_system.data.call_api import fetch_movie_data, fetch_user_data

DEFAULT_RATING = None
DEFAULT_WATCH_TIME = None

class DataPreprocessor:
    # Preprocesses Kafka events, validates schema, and updates user/movie CSVs

    def __init__(self, users_path, movies_path):
        self.users_path = users_path
        self.movies_path = movies_path
        self._ensure_csvs()

    def _ensure_csvs(self):
        # Create users and movies CSVs if they don't exist
        if not os.path.exists(self.users_path):
            pd.DataFrame(columns=[
                "user_id", "name", "age", "occupation", "gender", "join_date", "movies"
            ]).to_csv(self.users_path, index=False)

        if not os.path.exists(self.movies_path):
            pd.DataFrame(columns=[
                "movie_id", "tmdb_id", "imdb_id", "title", "original_title", "adult",
                "belongs_to_collection", "budget", "genres", "homepage", "original_language",
                "overview", "popularity", "poster_path", "production_companies",
                "production_countries", "release_date", "revenue", "runtime",
                "spoken_languages", "status", "vote_average", "vote_count"
            ]).to_csv(self.movies_path, index=False)

    def process_event(self, event_json):
        if isinstance(event_json, str):
            try:
                data = json.loads(event_json)
                is_json = True
            except json.JSONDecodeError:
                is_json = False
        else:
            data = event_json
            is_json = True

        if is_json and isinstance(data, dict):
            event_type = data.get("event_type")
            if event_type == "watch":
                event = WatchEvent(
                    timestamp=data["timestamp"],
                    user_id=data["user_id"],
                    movie_id=data["movie_id"],
                    minute=data["minute"],
                )
            elif event_type == "rate":
                event = RateEvent(
                    timestamp=data["timestamp"],
                    user_id=data["user_id"],
                    movie_id=data["movie_id"],
                    rating=data["rating"],
                )
            else:
                print(f"Skipping unsupported event type: {event_type}")
                return

        else:
            event = parse_event(event_json)

        if event is None:
            print(f"Skipping invalid event: {event_json}")
            return

        self._update_user_movie(event)
        self._update_movie_metadata(event.movie_id)

    def process_batch(self, event_json_list):
        users_df = pd.read_csv(self.users_path)

        new_users = set()
        seen_movies = set() 

        for event_json in event_json_list:
            if isinstance(event_json, str):
                try:
                    data = json.loads(event_json)
                except json.JSONDecodeError:
                    print(f"Skipping invalid event: {event_json}")
                    continue
            else:
                data = event_json

            event_type = data.get("event_type")
            if event_type not in {"watch", "rate"}:
                continue

            # Create event
            if event_type == "watch":
                event = WatchEvent(
                    timestamp=data["timestamp"],
                    user_id=data["user_id"],
                    movie_id=data["movie_id"],
                    minute=data["minute"]
                )
            elif event_type == "rate":
                event = RateEvent(
                    timestamp=data["timestamp"],
                    user_id=data["user_id"],
                    movie_id=data["movie_id"],
                    rating=data["rating"]
                )


            user_id, movie_id = event.user_id, event.movie_id
            seen_movies.add(movie_id) 

            # Ensure user exists
            if user_id not in users_df["user_id"].astype(str).values:
                new_users.add(user_id)
                users_df = pd.concat([users_df, pd.DataFrame([{
                    "user_id": user_id,
                    "name": f"User {user_id}",
                    "age": "",
                    "occupation": "",
                    "gender": "",
                    "join_date": datetime.now().strftime("%Y-%m-%d"),
                    "movies": json.dumps({})
                }])], ignore_index=True)

            idx = users_df.index[users_df["user_id"].astype(str) == user_id][0]
            movies = json.loads(users_df.at[idx, "movies"] or "{}")

            if movie_id not in movies:
                movies[movie_id] = {"rating": DEFAULT_RATING, "watch_time": DEFAULT_WATCH_TIME}

            if event_type == "watch":
                movies[movie_id]["watch_time"] = event.minute
            elif event_type == "rate":
                movies[movie_id]["rating"] = event.rating

            users_df.at[idx, "movies"] = json.dumps(movies)

        # Save updated user CSV
        users_df.to_csv(self.users_path, index=False)

        # Add new users via API
        for uid in new_users:
            self._add_new_user(uid)

        # Always ensure we have metadata for every seen movie
        for mid in seen_movies:
            self._update_movie_metadata(mid)

        print(f"Processed {len(event_json_list)} events, "
            f"{len(new_users)} new users, {len(seen_movies)} total movies")

    def _update_user_movie(self, event):
        # Update or add a single user entry
        users_df = pd.read_csv(self.users_path)
        user_id = event.user_id
        movie_id = event.movie_id

        if user_id not in users_df["user_id"].astype(str).values:
            self._add_new_user(user_id)
            users_df = pd.read_csv(self.users_path)

        idx = users_df.index[users_df["user_id"].astype(str) == user_id][0]
        raw_data = users_df.at[idx, "movies"]
        movies = json.loads(raw_data) if pd.notna(raw_data) and raw_data.strip() else {}

        if movie_id not in movies:
            movies[movie_id] = {"rating": DEFAULT_RATING, "watch_time": DEFAULT_WATCH_TIME}

        if isinstance(event, WatchEvent):
            movies[movie_id]["watch_time"] = event.minute
        elif isinstance(event, RateEvent):
            movies[movie_id]["rating"] = event.rating

        users_df.at[idx, "movies"] = json.dumps(movies)
        users_df.to_csv(self.users_path, index=False)

    def _add_new_user(self, user_id):
        # Fetch and add a new user entry from API
        metadata = fetch_user_data(user_id) or {}
        new_user = {
            "user_id": user_id,
            "name": metadata.get("name", f"User {user_id}"),
            "age": metadata.get("age", ""),
            "occupation": metadata.get("occupation", ""),
            "gender": metadata.get("gender", ""),
            "join_date": metadata.get("join_date", datetime.now().strftime("%Y-%m-%d")),
            "movies": json.dumps({})
        }

        users_df = pd.read_csv(self.users_path)
        users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
        users_df.to_csv(self.users_path, index=False)
        print(f"Added new user {user_id}")

    def _update_movie_metadata(self, movie_id):
        # Load current movies
        movies_df = pd.read_csv(self.movies_path)

        # Skip if already exists
        if movie_id in movies_df["movie_id"].astype(str).values:
            print(f"[DEBUG] Movie already exists in CSV: {movie_id}")
            return

        # Fetch metadata from API
        metadata = fetch_movie_data(movie_id)
        if not metadata:
            metadata = {"title": movie_id.replace("+", " ").title(), "genres": "Unknown"}

        # Build new entry (with placeholders for missing values)
        new_movie = {
            "movie_id": movie_id,
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
            "vote_count": metadata.get("vote_count", 0)
        }

        movies_df = pd.concat([movies_df, pd.DataFrame([new_movie])], ignore_index=True)
        movies_df.to_csv(self.movies_path, index=False)
        print(f"Added new movie: {movie_id}")
