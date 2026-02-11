import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
import pickle

from shrekommender_system.pipeline.pipeline import Pipeline

def make_dummy_artefacts():
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

@patch("shrekommender_system.recommenders.als_recommender.implicit.als.AlternatingLeastSquares")
@patch("shrekommender_system.data.ingestion.KafkaDataIngester.fetch_recent")
def test_pipeline_end_to_end_real_model(mock_ingest, mock_fit):
    mock_ingest.return_value = {"status": "completed", "files_created": ["dummy.jsonl"]}

    dummy_artefacts = make_dummy_artefacts()

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = Pipeline()
        pipeline.paths.data_root = Path(tmpdir) / "data"
        pipeline.paths.model_dir = Path(tmpdir) / "models"
        pipeline.paths.ensure_data_dirs("raw", "processed", "intermediate")

        with patch("shrekommender_system.recommenders.als_recommender.ALSRecommender.train") as mock_train:
            mock_train.side_effect = lambda config: setattr(pipeline.model, "train_matrix", dummy_artefacts["train_matrix"]) or \
                                                  setattr(pipeline.model, "user_mappings", dummy_artefacts["user_mappings"]) or \
                                                  setattr(pipeline.model, "movie_mappings", dummy_artefacts["movie_mappings"]) or \
                                                  setattr(pipeline.model, "most_popular_movies", np.array([0,1,2])) or \
                                                  setattr(pipeline.model, "is_loaded", True)

            stats = pipeline.run(n=2)

        assert stats["ingestion"]["status"] == "completed"
        assert stats["preprocessing"]["status"] == "completed"
        assert stats["model"]["status"] == "updated"
        model_path = Path(stats["model"]["model_path"])
        assert model_path.exists()

        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)
        assert "model" in saved_data
        assert "most_popular_movies" in saved_data
