import pickle
import numpy as np
from scipy.sparse import csr_matrix
from unittest.mock import patch
import tempfile
from pathlib import Path

import pytest
from shrekommender_system.recommenders.als_recommender import ALSRecommender, ModelNotLoadedException

@pytest.fixture
def dummy_artefacts():
    train_matrix = csr_matrix([[1,0,1],[0,1,0]])
    user_mappings = {"user_to_idx": {"u1": 0, "u2": 1}, "n_users": 2}
    movie_mappings = {
        "idx_to_movie": {0:"m1",1:"m2",2:"m3"},
        "movie_to_idx": {"m1":0,"m2":1,"m3":2},
        "n_movies": 3
    }
    return {
        "train_matrix": train_matrix,
        "user_mappings": user_mappings,
        "movie_mappings": movie_mappings
    }

@pytest.fixture
def als_model(dummy_artefacts):
    config = {"artefacts": dummy_artefacts, "hyperparameters": {"factors":2,"iterations":1}}
    als = ALSRecommender(config)
    with patch("implicit.als.AlternatingLeastSquares", autospec=True) as mock_als:
        mock_als.return_value.fit.return_value = None
        als.train(config)
    return als

def test_recommend_for_known_and_new_user(als_model):
    with patch.object(als_model.model, "recommend", return_value=(np.array([0,2]), np.array([0.9,0.8]))):
        rec_known = als_model.recommend_for_user("u1", N=2)
        assert rec_known.user_type == "known"
        assert rec_known.total_count == 2

    rec_new = als_model.recommend_for_user("new_user", N=2)
    assert rec_new.user_type == "new"
    assert rec_new.total_count == 2

def test_get_similar_items(als_model):
    with patch.object(als_model.model, "similar_items", return_value=(np.array([0,1,2]), np.array([1.0,0.8,0.5]))):
        result = als_model.get_similar_items("m1", N=2)
        assert result.item_id == "m1"
        assert result.total_count == 2
        assert all("movie_id" in rec for rec in result.similar_items)

    result_unknown = als_model.get_similar_items("mX")
    assert result_unknown.error is not None

def test_get_user_profile(als_model):
    profile_known = als_model.get_user_profile("u1")
    assert profile_known["user_type"] == "known"
    assert profile_known["interactions_count"] > 0

    profile_new = als_model.get_user_profile("unknown_user")
    assert profile_new["user_type"] == "new"
    assert profile_new["interactions_count"] == 0

def test_model_stats(als_model):
    stats = als_model.get_model_stats()
    assert stats["n_users"] == 2
    assert stats["n_movies"] == 3
    assert stats["matrix_shape"] == (2,3)

def test_save_model(als_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "als_model.pkl"
        als_model.save_model(tmpdir)
        assert model_path.exists()
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        assert "model" in data
        assert "most_popular_movies" in data

def test_model_not_loaded_exception(dummy_artefacts):
    config = {"artefacts": dummy_artefacts, "hyperparameters": {"factors":2,"iterations":1}}
    als = ALSRecommender(config)
    with pytest.raises(ModelNotLoadedException):
        als.recommend_for_user("u1")
