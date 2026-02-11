import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import argparse
# =========================
# Core drift utilities
# =========================

def _psi_from_series(ref: pd.Series, cur: pd.Series, bins=20) -> float:
    """PSI for numeric vectors."""
    r = ref.dropna().to_numpy()
    c = cur.dropna().to_numpy()
    if r.size == 0 or c.size == 0:
        return np.nan
    low, high = np.nanmin(r), np.nanmax(r)
    if low == high:
        # degenerate reference distribution: treat any shift as high PSI if cur differs
        return 0.0 if np.allclose(c, low) else 1.0
    r_hist, _ = np.histogram(r, bins=bins, range=(low, high), density=True)
    c_hist, _ = np.histogram(c, bins=bins, range=(low, high), density=True)
    r_hist = np.where(r_hist == 0, 1e-8, r_hist)
    c_hist = np.where(c_hist == 0, 1e-8, c_hist)
    return float(np.sum((c_hist - r_hist) * np.log(c_hist / r_hist)))

def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """PSI for categorical distributions (safe for high cardinality)."""
    r_counts = ref.dropna().astype(str).value_counts(normalize=True)
    c_counts = cur.dropna().astype(str).value_counts(normalize=True)
    keys = set(r_counts.index).union(c_counts.index)
    psi = 0.0
    for k in keys:
        p = r_counts.get(k, 1e-8)
        q = c_counts.get(k, 1e-8)
        psi += (q - p) * np.log(q / p)
    return float(psi)

def _ks_from_series(ref: pd.Series, cur: pd.Series):
    """KS statistic and p-value for numeric features."""
    r = ref.dropna().to_numpy()
    c = cur.dropna().to_numpy()
    if r.size == 0 or c.size == 0:
        return (np.nan, np.nan)
    stat, p = ks_2samp(r, c)
    return (float(stat), float(p))

def _severity_from(psi: float | None, ks_stat: float | None, ks_p: float | None) -> str:
    """
    PSI thresholds: <0.1 (low), 0.1–0.25 (med), >=0.25 (high)
    KS red if p<0.01 and D>0.1
    """
    sev = "low"
    if psi is not None and not np.isnan(psi):
        if psi >= 0.25:
            sev = "high"
        elif psi >= 0.10:
            sev = "medium"
    if ks_stat is not None and ks_p is not None and not (np.isnan(ks_stat) or np.isnan(ks_p)):
        if ks_p < 0.01 and ks_stat > 0.1:
            # escalate to high if KS strongly disagrees
            sev = "high"
    return sev

def generic_drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    bins: int = 20,
) -> pd.DataFrame:
    """Build a unified drift report for the provided column lists."""
    rows = []
    for col in numeric_cols:
        if col not in ref_df.columns or col not in cur_df.columns: 
            continue
        psi = _psi_from_series(ref_df[col], cur_df[col], bins=bins)
        ks_stat, ks_p = _ks_from_series(ref_df[col], cur_df[col])
        rows.append({
            "feature": col, "type": "numeric", "psi": psi,
            "ks_stat": ks_stat, "ks_p": ks_p,
            "severity": _severity_from(psi, ks_stat, ks_p),
            "ref_nonnull": int(ref_df[col].notna().sum()),
            "cur_nonnull": int(cur_df[col].notna().sum()),
        })
    for col in categorical_cols:
        if col not in ref_df.columns or col not in cur_df.columns: 
            continue
        psi = _psi_categorical(ref_df[col], cur_df[col])
        rows.append({
            "feature": col, "type": "categorical", "psi": psi,
            "ks_stat": np.nan, "ks_p": np.nan,
            "severity": _severity_from(psi, None, None),
            "ref_nonnull": int(ref_df[col].notna().sum()),
            "cur_nonnull": int(cur_df[col].notna().sum()),
        })
    return pd.DataFrame(rows).sort_values(["severity","feature"], ascending=[False, True]).reset_index(drop=True)

# =========================
# Dataset: MOVIES
# =========================

def _safe_json_load(x):
    if pd.isna(x) or x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x)
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = s.replace("'", '"').replace("True","true").replace("False","false").replace("None","null")
            return json.loads(s2)
        except Exception:
            return None

def preprocess_movies_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # parse json-like columns if present
    for col in ["belongs_to_collection","genres","production_companies","production_countries","spoken_languages"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_json_load)
    # coerce types on key numerics (won't fail if col missing)
    for col in ["budget","revenue","runtime","vote_average","vote_count","tmdb_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        # add handy derived: release_year (categorical/numeric small range)
        df["release_year"] = df["release_date"].dt.year
    return df

def movies_drift_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> pd.DataFrame:
    ref = preprocess_movies_for_drift(ref_df)
    cur = preprocess_movies_for_drift(cur_df)

    numeric_cols = [c for c in ["budget","revenue","runtime","vote_average","vote_count","release_year"] if c in ref.columns and c in cur.columns]
    categorical_cols = [c for c in ["status","adult"] if c in ref.columns and c in cur.columns]
    # You can also add imdb_id patterns, but those are high-cardinality IDs (not ideal for drift).

    return generic_drift_report(ref, cur, numeric_cols, categorical_cols, bins=20)

# =========================
# Dataset: USER EVENTS
# =========================

def preprocess_events_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["event_hour"] = df["timestamp"].dt.hour
        df["event_dow"]  = df["timestamp"].dt.dayofweek  # 0=Mon
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "user_id" in df.columns:
        # Keep user_id as string to avoid fake numerics; categorical drift can still be informative at coarse level
        df["user_id"] = df["user_id"].astype(str)
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    if "movie_id" in df.columns:
        df["movie_id"] = df["movie_id"].astype(str)
    return df

def events_drift_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> pd.DataFrame:
    ref = preprocess_events_for_drift(ref_df)
    cur = preprocess_events_for_drift(cur_df)

    numeric_cols = [c for c in ["rating","event_hour","event_dow"] if c in ref.columns and c in cur.columns]
    # Categorical: event_type; (optionally) coarse user_id/movie_id buckets are usually too high-cardinality; keep event_type by default
    categorical_cols = [c for c in ["event_type"] if c in ref.columns and c in cur.columns]

    return generic_drift_report(ref, cur, numeric_cols, categorical_cols, bins=10)

# ---------- Users dataset: preprocess + drift report ----------

def preprocess_users_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure columns exist (optional, keeps pipeline resilient)
    for col in ["user_id", "age", "occupation", "gender"]:
        if col not in df.columns:
            df[col] = pd.NA

    # IDs as strings (generally avoid drift on raw IDs)
    df["user_id"] = df["user_id"].astype(str).str.strip()

    # Age numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Normalize gender to a compact set
    def _norm_gender(x):
        s = (str(x) if pd.notna(x) else "").strip().lower()
        if s in {"f","female"}: return "F"
        if s in {"m","male"}:   return "M"
        if s in {"nonbinary","non-binary","nb","enby","x","other"}: return "Other"
        return "Unspecified"

    df["gender"] = df["gender"].apply(_norm_gender)

    # Normalize occupation (lowercased, trimmed; unify common “unspecified” variants)
    def _norm_occ(x):
        if pd.isna(x): return "unspecified"
        s = str(x).strip().lower()
        if s in {"", "none", "n/a", "na", "other or not specified", "unspecified", "unknown"}:
            return "unspecified"
        return s

    df["occupation"] = df["occupation"].apply(_norm_occ)

    # Optional: create an age bin feature (categorical) if you like categorical PSI for age too
    # df["age_bin"] = pd.cut(df["age"], bins=[-1,17,24,34,44,54,64,120],
    #                        labels=["0-17","18-24","25-34","35-44","45-54","55-64","65+"])

    return df

def users_drift_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build PSI + KS drift report for users dataset.

    Numeric:  age  (PSI + KS)
    Categorical: gender, occupation  (PSI)
    """
    ref = preprocess_users_for_drift(ref_df)
    cur = preprocess_users_for_drift(cur_df)

    numeric_cols = [c for c in ["age"] if c in ref.columns and c in cur.columns]
    categorical_cols = [c for c in ["gender", "occupation"] if c in ref.columns and c in cur.columns]

    # If you enabled age_bin above, you can also append "age_bin" to categorical_cols.
    # categorical_cols.append("age_bin")

    return generic_drift_report(ref, cur, numeric_cols, categorical_cols, bins=10)

parser = argparse.ArgumentParser(description ='Process some integers.')
parser.add_argument('--ref', required=True, type = str, help ='current location of .csv containing old data (data used as refrence)')

parser.add_argument('--new', required=True, type = str, help ='current location of .csv containing new data (data collected from kafka stream (not used in training))')

parser.add_argument('--type', required=True, type = str, help ='type of data (user, movie, rating)')
args = parser.parse_args()


if __name__ == '__main__':
    df_ref=pd.read_csv(args.ref)
    df_new=pd.read_csv(args.new)
    if args.type == "user":
        print(users_drift_report(df_ref, df_new))
    elif args.type == "movie":
        print(movies_drift_report(df_ref, df_new))
    elif args.type == "rating":
        print(events_drift_report(df_ref, df_new))
    else:
        raise ValueError(f'Type {args.type} is not supported. Supported types are (user, movie, rating)')


    