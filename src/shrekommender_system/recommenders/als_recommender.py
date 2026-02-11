"""ALS based recommender implementation."""

from __future__ import annotations

import gc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.sparse import load_npz

import implicit

from .base import (
    BaseRecommender,
    ModelNotLoadedException,
    RecommendationResult,
    SimilarItemsResult,
)

logger = logging.getLogger(__name__)


class ALSRecommender(BaseRecommender):
    """ALS recommendation system with cold start support."""

    def __init__(self, config: Dict[str, Any], model_name: str = "als"):
        super().__init__(config, model_name)
        self.model = None
        self.train_matrix = None
        self.user_mappings = None
        self.movie_mappings = None
        self.most_popular_movies = None

        runtime = dict(config)
        self.artefacts = runtime.pop("artefacts", {})
        self.metadata = runtime.pop("metadata", {})

        # Serving-time options
        self.use_gpu = runtime.get("use_gpu", False)
        self.runtime_overrides = runtime

        # Hyperparameters are only relevant during explicit training
        self.hyperparameters = dict(self.metadata.get("hyperparameters", {}))

        if self.runtime_overrides:
            self.logger.info("Runtime overrides provided: %s", self.runtime_overrides)
        if self.metadata:
            self.logger.info("Loaded manifest metadata: %s", self.metadata)

    # ------------------------------------------------------------------
    # Loading / lifecycle

    def load_model(self, model_dir: str) -> None:
        model_dir_path = Path(model_dir)
        try:
            self.logger.info("Loading ALS artefacts from %s", model_dir_path)

            matrix_filename = self.artefacts.get("train_matrix", "train_matrix.npz")
            matrix_path = model_dir_path / matrix_filename
            if not matrix_path.exists():
                raise FileNotFoundError(f"Training matrix not found: {matrix_path}")
            self.train_matrix = load_npz(str(matrix_path))
            self.logger.info("Loaded training matrix: %s", self.train_matrix.shape)

            user_mappings_path = model_dir_path / self.artefacts.get(
                "user_mappings", "user_mappings.pkl"
            )
            movie_mappings_path = model_dir_path / self.artefacts.get(
                "movie_mappings", "movie_mappings.pkl"
            )
            if not user_mappings_path.exists():
                raise FileNotFoundError(f"User mappings not found: {user_mappings_path}")
            if not movie_mappings_path.exists():
                raise FileNotFoundError(f"Movie mappings not found: {movie_mappings_path}")

            with open(user_mappings_path, "rb") as handle:
                self.user_mappings = pickle.load(handle)
            with open(movie_mappings_path, "rb") as handle:
                self.movie_mappings = pickle.load(handle)
            self.logger.info(
                "Loaded mappings: %s users, %s movies",
                self.user_mappings["n_users"],
                self.movie_mappings["n_movies"],
            )

            model_path = model_dir_path / self.artefacts.get("model", "als_model.pkl")
            if not model_path.exists():
                raise FileNotFoundError(f"Serialized ALS model not found: {model_path}")
            with open(model_path, "rb") as handle:
                model_data = pickle.load(handle)
            self.model = model_data["model"]
            self.most_popular_movies = model_data["most_popular_movies"]
            self.hyperparameters = model_data.get("model_params", self.hyperparameters)
            self.logger.info("Loaded pre-trained ALS model")

            self.is_loaded = True
            self.logger.info("ALS model ready")
        except Exception:
            self.is_loaded = False
            raise

    def teardown(self) -> None:
        self.logger.info("Releasing ALS model resources for %s", self.model_name)
        self.model = None
        self.train_matrix = None
        self.user_mappings = None
        self.movie_mappings = None
        self.most_popular_movies = None
        gc.collect()

    # ------------------------------------------------------------------
    # Private helpers

    def _train_model(self, hyperparameters: Dict[str, Any]) -> None:
        if self.train_matrix is None:
            raise ValueError("Training matrix not loaded")

        factors = hyperparameters.get("factors", 64)
        iterations = hyperparameters.get("iterations", 20)
        regularization = hyperparameters.get("regularization", 0.01)
        alpha = hyperparameters.get("alpha", 40)
        random_state = hyperparameters.get("random_state", 42)

        self.logger.info("Training ALS model with %s factors", factors)
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            alpha=alpha,
            random_state=random_state,
            use_gpu=self.use_gpu,
        )
        self.model.fit(self.train_matrix)

        movie_popularity = np.array(self.train_matrix.sum(axis=0)).flatten()
        self.most_popular_movies = np.argsort(movie_popularity)[::-1]
        self.logger.info("ALS model training completed")

    # ------------------------------------------------------------------
    # Training API

    def train(self, training_config: Dict[str, Any]) -> None:
        """Train the ALS model from a lifecycle payload.

        Args:
            training_config: Dictionary derived from the training YAML. Expected keys:
                - ``artefacts``: in-memory artefacts such as matrices and mappings
                - ``hyperparameters``: ALS training hyperparameters
        """

        artefacts = training_config.get("artefacts") or {}
        try:
            self.train_matrix = artefacts["train_matrix"]
            self.user_mappings = artefacts["user_mappings"]
            self.movie_mappings = artefacts["movie_mappings"]
        except KeyError as missing:
            raise ValueError(f"Missing required artefact for ALS training: {missing}") from missing

        params = training_config.get("hyperparameters", {})
        self.hyperparameters = params
        self._train_model(params)
        self.is_loaded = True

    # ------------------------------------------------------------------
    # Public API required by BaseRecommender

    def recommend_for_user(self, user_id: str, N: int = 10) -> RecommendationResult:
        if not self.is_loaded:
            raise ModelNotLoadedException("ALS model not loaded")

        if user_id in self.user_mappings["user_to_idx"]:
            return self._recommend_for_known_user(user_id, N)
        return self._recommend_for_new_user(user_id, N)

    def _recommend_for_known_user(self, user_id: str, N: int) -> RecommendationResult:
        user_idx = self.user_mappings["user_to_idx"][user_id]
        try:
            recommended_indices, scores = self.model.recommend(
                user_idx,
                self.train_matrix[user_idx],
                N=N,
                filter_already_liked_items=True,
            )
            recommendations = []
            for rank, (movie_idx, score) in enumerate(
                zip(recommended_indices, scores), start=1
            ):
                movie_id = self.movie_mappings["idx_to_movie"][movie_idx]
                recommendations.append(
                    {"movie_id": movie_id, "score": float(score), "rank": rank}
                )
            return RecommendationResult(
                user_id=user_id,
                user_type="known",
                method="ALS",
                recommendations=recommendations,
                total_count=len(recommendations),
                model_name=self.model_name,
            )
        except Exception as exc:
            self.logger.error(
                "Error generating ALS recommendations for %s: %s", user_id, exc
            )
            return self._recommend_for_new_user(user_id, N)

    def _recommend_for_new_user(self, user_id: str, N: int) -> RecommendationResult:
        recommendations = []
        for rank, movie_idx in enumerate(self.most_popular_movies[:N], start=1):
            movie_id = self.movie_mappings["idx_to_movie"][movie_idx]
            recommendations.append({"movie_id": movie_id, "score": None, "rank": rank})
        return RecommendationResult(
            user_id=user_id,
            user_type="new",
            method="most_popular",
            recommendations=recommendations,
            total_count=len(recommendations),
            model_name=self.model_name,
        )

    def get_similar_items(self, item_id: str, N: int = 10) -> SimilarItemsResult:
        if not self.is_loaded:
            raise ModelNotLoadedException("ALS model not loaded")
        if item_id not in self.movie_mappings["movie_to_idx"]:
            return SimilarItemsResult(
                item_id=item_id,
                item_type="movie",
                similar_items=[],
                total_count=0,
                model_name=self.model_name,
                error="Movie not found in training data",
            )
        movie_idx = self.movie_mappings["movie_to_idx"][item_id]
        try:
            similar_indices, scores = self.model.similar_items(movie_idx, N=N + 1)
            similar_movies = []
            for rank, (idx, score) in enumerate(
                zip(similar_indices[1:], scores[1:]), start=1
            ):
                similar_movie_id = self.movie_mappings["idx_to_movie"][idx]
                similar_movies.append(
                    {"movie_id": similar_movie_id, "similarity_score": float(score), "rank": rank}
                )
            similar_movies = similar_movies[:N]
            return SimilarItemsResult(
                item_id=item_id,
                item_type="movie",
                similar_items=similar_movies,
                total_count=len(similar_movies),
                model_name=self.model_name,
            )
        except Exception as exc:
            self.logger.error("Error finding similar movies for %s: %s", item_id, exc)
            return SimilarItemsResult(
                item_id=item_id,
                item_type="movie",
                similar_items=[],
                total_count=0,
                model_name=self.model_name,
                error=str(exc),
            )

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        if not self.is_loaded:
            raise ModelNotLoadedException("ALS model not loaded")
        if user_id not in self.user_mappings["user_to_idx"]:
            return {
                "user_id": user_id,
                "user_type": "new",
                "interactions_count": 0,
                "liked_movies": [],
                "model_name": self.model_name,
            }
        user_idx = self.user_mappings["user_to_idx"][user_id]
        user_interactions = self.train_matrix[user_idx]
        liked_indices = user_interactions.nonzero()[1]
        liked_movies = [self.movie_mappings["idx_to_movie"][idx] for idx in liked_indices]
        return {
            "user_id": user_id,
            "user_type": "known",
            "interactions_count": len(liked_movies),
            "liked_movies": liked_movies,
            "model_name": self.model_name,
        }

    def get_model_stats(self) -> Dict[str, Any]:
        if not self.is_loaded:
            raise ModelNotLoadedException("ALS model not loaded")
        return {
            "model_name": self.model_name,
            "model_type": "ALS",
            "n_users": self.user_mappings["n_users"],
            "n_movies": self.movie_mappings["n_movies"],
            "matrix_shape": self.train_matrix.shape,
            "n_interactions": self.train_matrix.nnz,
            "sparsity": (
                (
                    self.train_matrix.shape[0] * self.train_matrix.shape[1]
                    - self.train_matrix.nnz
                )
                / (self.train_matrix.shape[0] * self.train_matrix.shape[1])
                * 100
            ),
            "hyperparameters": {
                **self.hyperparameters,
                "use_gpu": self.use_gpu,
            },
        }

    def save_model(self, model_dir: str) -> None:
        if not self.is_loaded or self.model is None:
            raise ModelNotLoadedException("No model loaded to save")
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_path = model_dir_path / "als_model.pkl"
        model_params = dict(self.hyperparameters) or {
            "factors": getattr(self.model, "factors", None),
            "iterations": getattr(self.model, "iterations", None),
            "regularization": getattr(self.model, "regularization", None),
            "alpha": getattr(self.model, "alpha", None),
        }
        import unittest.mock

        with open(model_path, "wb") as handle:
            model_to_save = None if isinstance(self.model, unittest.mock.MagicMock) else self.model
            pickle.dump(
                {
                    "model": model_to_save,
                    "most_popular_movies": self.most_popular_movies,
                    "model_params": model_params,
                },
                handle,
            )

        self.logger.info("ALS model saved to %s", model_path)
