import json, re, pandas as pd, numpy as np
import pandera.pandas as pa
from pandera import Column, Check
import argparse
#------------------Events------------------
def preprocess_events_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure columns exist (optional)
    expected = {"timestamp","user_id","event_type","movie_id","rating"}
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce obvious types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce", downcast="integer")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")  # allow float/int
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df["movie_id"] = df["movie_id"].astype(str).str.strip()

    return df

# Allowed event types — extend as your product needs
EVENT_TYPES = {"view", "rate", "wishlist", "click", "play", "stop"}

# --- Schema ------------------------------------------------------------------
EventsSchema = pa.DataFrameSchema(
    {
        "timestamp": Column(
            pa.DateTime, nullable=False,
            checks=[
                # sanity window; adjust as needed
                Check(lambda s: s.dt.year.between(2000, 2100)),
            ],
        ),
        "user_id": Column(int, nullable=False, checks=Check.ge(1)),
        "event_type": Column(pa.String, nullable=False, checks=Check.isin(EVENT_TYPES)),
        "movie_id": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
        # rating may be null (for non-rate events); bounds enforced conditionally below
        "rating": Column(float, nullable=True),
    },
    coerce=True,
    checks=[
        # If event_type == "rate" -> rating must be notna and between 0..10
        Check(
            lambda df: (
                (~(df["event_type"] == "rate")) |  # not a rating event -> ok regardless
                (
                    df["rating"].notna() &
                    (df["rating"] >= 0.0) &
                    (df["rating"] <= 10.0)
                )
            )
        ),
        # If event_type != "rate" -> rating should be NA (optional; remove if you want to keep stray numbers)
        Check(
            lambda df: (
                (df["event_type"] == "rate") |
                (df["rating"].isna())
            )
        ),
    ],
)

# --- Validate + drop bad rows ------------------------------------------------
def validate_events_drop(df: pd.DataFrame):
    """
    Clean, validate, and DROP rows that don't conform.
    Returns (clean_df, report_dict).
    """
    original_n = len(df)
    df = preprocess_events_df(df)

    bad_idx = set()
    try:
        EventsSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        if "index" in err.failure_cases.columns:
            bad_idx.update(err.failure_cases["index"].dropna().astype(int).tolist())

    if bad_idx:
        df = df.drop(index=list(bad_idx), errors="ignore")

    if not df.empty:
        df = EventsSchema.validate(df, lazy=False)

    report = {
        "rows_in": original_n,
        "dropped_schema": len(bad_idx),
        "rows_out": len(df),
    }
    return df.reset_index(drop=True), report

#---------------------Movies-----------------------------------------

JSON_COLS = [
    "belongs_to_collection",
    "genres",
    "production_companies",
    "production_countries",
    "spoken_languages",
]


def _safe_json_load(x):
    if pd.isna(x) or x == "" or x is None:
        return None
    if isinstance(x, (dict, list)):  # already parsed
        return x
    s = str(x)
    # Some CSVs contain single quotes; try to normalize to valid JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            # naive fixes: single→double quotes, True/False/None→lowercase
            s2 = (
                s.replace("'", '"')
                 .replace("True", "true")
                 .replace("False", "false")
                 .replace("None", "null")
            )
            return json.loads(s2)
        except Exception:
            return None  # mark unparsable as None (will fail schema checks)
def preprocess_movies_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse JSON-like columns and do light coercions."""
    df = df.copy()

    # Ensure all expected columns exist (optional)
    expected = {
        "id","tmdb_id","imdb_id","title","original_title","adult",
        "belongs_to_collection","budget","genres","homepage","poster_path",
        "production_companies","production_countries","release_date","revenue",
        "runtime","spoken_languages","status","vote_average","vote_count",
    }
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    # Parse JSON-like columns
    for col in JSON_COLS:
        df[col] = df[col].apply(_safe_json_load)

    # Coerce some common types ahead of schema
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce")
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
    # booleans sometimes come as strings
    df["adult"] = df["adult"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    # dates
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", utc=False)

    return df


def _is_list_of_dicts_or_none(val):
    return (val is None) or (isinstance(val, list) and all(isinstance(v, dict) for v in val))

def _is_dict_or_none(val):
    return (val is None) or isinstance(val, dict)


MoviesSchema = pa.DataFrameSchema(
    {
        # IDs & titles
        "id": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
        "tmdb_id": Column(int, nullable=False, checks=Check.ge(1)),
        "imdb_id": Column(
            pa.String,
            nullable=False,
            checks=[
                Check.str_matches(r"^tt\d{7,9}$"),  # e.g., tt0330373
            ],
        ),
        "title": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
        "original_title": Column(pa.String, nullable=True),

        # Flags & money
        "adult": Column(pa.Bool, nullable=False),
        "budget": Column(int, nullable=True, checks=Check.ge(0)),
        "revenue": Column(int, nullable=True, checks=Check.ge(0)),

        # JSON-like fields (already parsed in preprocess)
        "belongs_to_collection": Column(
            object, nullable=True, checks=Check(lambda s: s.map(_is_dict_or_none).all())
        ),
        "genres": Column(
            object, nullable=True, checks=Check(lambda s: s.map(_is_list_of_dicts_or_none).all())
        ),
        "production_companies": Column(
            object, nullable=True, checks=Check(lambda s: s.map(_is_list_of_dicts_or_none).all())
        ),
        "production_countries": Column(
            object, nullable=True, checks=Check(lambda s: s.map(_is_list_of_dicts_or_none).all())
        ),
        "spoken_languages": Column(
            object, nullable=True, checks=Check(lambda s: s.map(_is_list_of_dicts_or_none).all())
        ),

        # URLs/paths
        "homepage": Column(pa.String, nullable=True, checks=Check(lambda s: s.fillna("").str.len() <= 2048)),
        "poster_path": Column(pa.String, nullable=True),

        # Dates & numerics
        "release_date": Column(
            pa.DateTime,  # pandas.Timestamp
            nullable=True,
            checks=[
                Check(lambda s: s.isna() | (s.dt.year.between(1880, 2100))),
            ],
        ),
        "runtime": Column(int, nullable=True, checks=Check(lambda s: s.isna() | s.between(0, 2000))),
        "vote_average": Column(float, nullable=True, checks=Check(lambda s: s.isna() | (s.between(0, 10)))),
        "vote_count": Column(int, nullable=True, checks=Check(lambda s: s.isna() | (s >= 0))),

        # Status enum (use what you actually see in your data)
        "status": Column(
            pa.String, nullable=True,
            checks=Check.isin({"Rumored","Planned","In Production","Post Production","Released","Canceled"})
        ),
    },
    # coerce=True,  # let Pandera coerce where possible
)

def validate_movies_drop(df: pd.DataFrame):
    """
    Preprocess JSON-like fields, validate schema, and drop any non-conforming rows.
    Returns (clean_df, report).
    """
    original_n = len(df)
    df = preprocess_movies_df(df)

    bad_idx = set()
    try:
        MoviesSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        if "index" in err.failure_cases.columns:
            bad_idx.update(
                err.failure_cases["index"].dropna().astype(int).tolist()
            )
            print(err)

    if bad_idx:
        df = df.drop(index=list(bad_idx), errors="ignore")

    if not df.empty:
        df = MoviesSchema.validate(df, lazy=False)

    report = {
        "rows_in": original_n,
        "dropped_schema": len(bad_idx),
        "rows_out": len(df),
    }
    return df.reset_index(drop=True), report

#---------------------Users -----------------------

UIsersSchema = pa.DataFrameSchema({
    "user_id": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
    "age": Column(int, nullable=False, checks=[Check.ge(5), Check.le(100)]),
    "occupation": Column(pa.String, nullable=False, checks=[Check.str_length(min_value=1) , Check.isin({'other or not specified', 'executive/managerial',
       'sales/marketing', 'college/grad student', 'doctor/health care',
       'academic/educator', 'homemaker', 'K-12 student', 'self-employed',
       'scientist', 'technician/engineer', 'clerical/admin', 'artist',
       'tradesman/craftsman', 'retired', 'unemployed', 'programmer',
       'customer service', 'writer', 'lawyer', 'farmer'})]),
    "gender": Column(pa.String, nullable=False, checks=[Check.str_length(max_value=1) ,Check.isin({'F', 'M'})]),
})

def validate_users_drop(df: pd.DataFrame):
    """
    Preprocess JSON-like fields, validate schema, and drop any non-conforming rows.
    Returns (clean_df, report).
    """
    original_n = len(df)
    
    bad_idx = set()
    try:
        UIsersSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        if "index" in err.failure_cases.columns:
            bad_idx.update(
                err.failure_cases["index"].dropna().astype(int).tolist()
            )
            print(err)

    if bad_idx:
        df = df.drop(index=list(bad_idx), errors="ignore")

    if not df.empty:
        df = UIsersSchema.validate(df, lazy=False)

    report = {
        "rows_in": original_n,
        "dropped_schema": len(bad_idx),
        "rows_out": len(df),
    }
    return df.reset_index(drop=True), report

parser = argparse.ArgumentParser(description ='Process some integers.')
parser.add_argument('--df', required=True, type = str, help ='current location of .csv file to check schema)')

parser.add_argument('--type', required=True, type = str, help ='type of data (user, movie, rating)')
args = parser.parse_args()

if __name__ == "__main__":
    df=pd.read_csv(args.df)
    if args.type == "user":
        print(validate_users_drop(df))
    elif args.type == "movie":
        print(validate_movies_drop(df))
    elif args.type == "rating":
        print(validate_events_drop(df))
    else:
        raise ValueError(f'Type {args.type} is not supported. Supported types are (user, movie, rating)')

