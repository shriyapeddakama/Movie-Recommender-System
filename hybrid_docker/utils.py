import ast
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
import pandas as pd
import time
def item_feature_tokens(row):
    tok = []
    if len(row["genre_names"]):
        tok += [f"genre:{g}" for g in row["genre_names"]]
    if len(row["spoken_languages_names"]):
        tok += [f"lang:{g}" for g in row["spoken_languages_names"]]
    if len(row["production_countries_names"]):
        tok += [f"country:{g}" for g in row["production_countries_names"]]
    if len(row["production_companies_names"]):
        tok += [f"comp:{g}" for g in row["production_companies_names"]]

    return tok

def user_feature_tokens(row):
    tok = []
    if len(row["age_bucket"]):
        tok.append(f"age:{row['age_bucket']}")
    if len(row["occupation"]):
        tok.append(f"job:{row['occupation']}")
    if len(row["gender"]):
        tok.append(f"gender:{row['gender']}")
        

    return tok

def parse_json(val):
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, str):
        try:
            lst = ast.literal_eval(val)
        except Exception:
            return []
    else:
        lst = val
    return [d.get("name") for d in lst if isinstance(d, dict) and "name" in d]

def convert_columns(df_user, df_movie):
    df_movie['belongs_to_collection']= df_movie['belongs_to_collection'].apply(lambda x : 'False' if x == '{}' else 'True')
    df_movie['adult']= df_movie['adult'].apply(lambda x : 'True' if x else 'False')
    age_bins = [0, 17, 24, 34, 44, 54, 64, 120]
    age_labels = ["0-17","18-24","25-34","35-44","45-54","55-64","65+"]
    df_user["age_bucket"] = pd.cut(df_user["age"], bins=age_bins, labels=age_labels, right=True)


    df_movie["genre_names"] = df_movie["genres"].apply(parse_json)
    df_movie["production_companies_names"] = df_movie["production_companies"].apply(parse_json)
    df_movie["production_countries_names"] = df_movie["production_countries"].apply(parse_json)
    df_movie["spoken_languages_names"] = df_movie["spoken_languages"].apply(parse_json)

    return df_user, df_movie

def get_names(df_user, df_movie):
    all_user_ids = df_user["user_id"].astype(str).tolist()
    all_item_ids = df_movie["id"].astype(str).tolist()
    all_user_feats = sorted({t for tl in df_user["item_tokens"] for t in tl})
    all_item_feats = sorted({t for tl in df_movie["item_tokens"] for t in tl})
    return all_user_ids, all_item_ids, all_user_feats, all_item_feats

def process_data(df_user, df_movie, df_rating):
    df_user, df_movie= convert_columns(df_user, df_movie)
    df_movie["item_tokens"] = df_movie.apply(item_feature_tokens, axis=1)
    df_user["item_tokens"] = df_user.apply(user_feature_tokens, axis=1)
    all_user_ids, all_item_ids, all_user_feats, all_item_feats= get_names(df_user, df_movie)

    user_feats_iter = (
    (str(row.user_id), row.item_tokens)
    for row in df_user[['user_id','item_tokens']].itertuples(index=False)
    )
    item_feats_iter = (
        (str(row.id), row.item_tokens)
        for row in df_movie[['id','item_tokens']].itertuples(index=False)
    )
    triples = df_rating[['user_id','movie_id']].astype({'user_id': str, 'movie_id': str}).itertuples(index=False, name=None)

    return all_user_ids, all_item_ids, all_user_feats, all_item_feats, user_feats_iter, item_feats_iter, triples

def split_train_test(df_user, df_rating, frac=0.1):
    test_users= df_user.sample(frac=frac, replace=False).copy()
    df_user.drop(test_users.index, inplace= True)

    train_rating = df_rating[df_rating['user_id'].isin(df_user['user_id'].tolist())].copy()
    test_rating = df_rating[df_rating['user_id'].isin(test_users['user_id'].tolist())].copy()

    test_users.reset_index(inplace=True, drop=True)
    df_user.reset_index(inplace=True, drop=True)
    train_rating.reset_index(inplace=True, drop=True)
    test_rating.reset_index(inplace=True, drop=True)
    return df_user, train_rating, test_users, test_rating

def train_model(model, train_data, test_data, features, epoch):
    ITEM_ALPHA = 1e-6
    for ep in range(1, epoch + 1):
        start_train= time.time()
        # train exactly 1 epoch
        model.fit_partial(
            train_data,
            #sample_weight=features[0],           # or None
            user_features=features[1],     # or None
            item_features=features[2],     # or None
            epochs=1,
            num_threads=4
        )
        end_train= time.time()
        start_eval= time.time()
        # compute metrics on the training set (or a held-out test set if you have one)
        test_auc = auc_score(
            model, test_data,
            user_features=features[1],
            item_features=features[2],
            num_threads=4,
            check_intersections=False
        ).mean()

        test_p10 = precision_at_k(
            model, test_data, k=20,
            user_features=features[1],
            item_features=features[2],
            num_threads=4,
            check_intersections=False
        ).mean()

        test_k10 = recall_at_k(
            model, test_data, k=20,
            user_features=features[1],
            item_features=features[2],
            num_threads=4,
            check_intersections=False
        ).mean()

        #####
        train_auc = auc_score(
            model, train_data,
            user_features=features[1],
            item_features=features[2],
            num_threads=4
        ).mean()

        train_p10 = precision_at_k(
            model, train_data, k=20,
            user_features=features[1],
            item_features=features[2],
            num_threads=4
        ).mean()

        train_k10 = recall_at_k(
            model, train_data, k=20,
            user_features=features[1],
            item_features=features[2],
            num_threads=4
        ).mean()
        end_eval= time.time()

        print(f"Epoch {ep}/{epoch} - Test (AUC: {test_auc:.4f}  Precision@10: {test_p10:.4f}  Recall@10: {test_k10:.4f})\n\tTrain (AUC: {train_auc:.4f}  Precision@10: {train_p10:.4f}  Recall@10: {train_k10:.4f})")
        print(f"train time: {end_train-start_train}s\teval time: {end_eval-start_eval}s")
    return model
