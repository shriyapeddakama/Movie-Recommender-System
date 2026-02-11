import numpy as np
import joblib
from lightfm import LightFM
import scipy.sparse as sp
def build_coldstart_user_row(dataset, tokens):
    """
    Build a 1 x n_user_feature_cols CSR aligned to the training vocab.
    'tokens' are the same kind of strings you used at training time (e.g., 'age:25-34').
    """
    # LightFM stores feature mappings internally; we use them to align columns.
    fmap = dataset._user_feature_mapping   # token -> col index (private API)
    cols = [fmap[t] for t in tokens if t in fmap]
    data = [1.0] * len(cols)
    return sp.csr_matrix((data, ([0]*len(cols), cols)),
                         shape=(1, len(fmap)))

model = joblib.load("/home/hbukhari/Project/lightfm_model.joblib")
dataset = joblib.load("/home/hbukhari/Project/lightfm_dataset.joblib")
user_features = joblib.load("/home/hbukhari/Project/user_features.joblib")
item_features = joblib.load("/home/hbukhari/Project/item_features.joblib")

U_row = build_coldstart_user_row(dataset, ["age:25-34", "job:engineer", "gender:M"])
n_items = dataset._item_feature_mapping.shape[0] 
item_ids = np.arange(n_items, dtype=np.int32)

scores = model.predict(
    user_ids=np.zeros(n_items, dtype=np.int32),   # all reference row 0 of U_row
    item_ids=item_ids,
    user_features=U_row,                           # 1×F_u matrix we just built
    item_features=item_features,                  # training item_features
    num_threads=4
)

topN_idx = scores.argsort()[::-1][:10]
# map internal indices -> your external item IDs
inv_item_map = {v: k for k, v in dataset._item_id_mapping.items()}
top_items = [inv_item_map[i] for i in topN_idx]

print('recommendations:', top_items)
