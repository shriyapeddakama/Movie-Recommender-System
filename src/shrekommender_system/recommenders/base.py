"""Core recommender abstractions used by the serving layer."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Standardised recommendation result returned by models."""

    user_id: str
    user_type: str
    method: str
    recommendations: List[Dict[str, Any]]
    total_count: int
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_type": self.user_type,
            "method": self.method,
            "recommendations": self.recommendations,
            "total_count": self.total_count,
            "model_name": self.model_name,
        }


@dataclass
class SimilarItemsResult:
    """Output structure for similar-item queries."""

    item_id: str
    item_type: str
    similar_items: List[Dict[str, Any]]
    total_count: int
    model_name: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "similar_items": self.similar_items,
            "total_count": self.total_count,
            "model_name": self.model_name,
        }
        if self.error:
            result["error"] = self.error
        return result


class BaseRecommender(ABC):
    """Base class for all recommenders loaded by the registry."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.is_loaded = False
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

    @abstractmethod
    def load_model(self, model_dir: str) -> None:
        """Load the model artefacts located in *model_dir*."""

    @abstractmethod
    def recommend_for_user(self, user_id: str, N: int = 10) -> RecommendationResult:
        """Return personalised recommendations for *user_id*."""

    @abstractmethod
    def get_similar_items(self, item_id: str, N: int = 10) -> SimilarItemsResult:
        """Return similar items for *item_id*."""

    @abstractmethod
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Return profile information describing *user_id*."""

    @abstractmethod
    def get_model_stats(self) -> Dict[str, Any]:
        """Return diagnostics describing the loaded model instance."""

    def health_check(self) -> Dict[str, Any]:
        try:
            if not self.is_loaded:
                return {
                    "model_name": self.model_name,
                    "status": "unhealthy",
                    "reason": "Model not loaded",
                }
            stats = self.get_model_stats()
            if not stats:
                return {
                    "model_name": self.model_name,
                    "status": "unhealthy",
                    "reason": "Failed to fetch stats",
                }
            return {
                "model_name": self.model_name,
                "status": "healthy",
                "stats": stats,
            }
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Health check failed for %s: %s", self.model_name, exc)
            return {
                "model_name": self.model_name,
                "status": "unhealthy",
                "reason": str(exc),
            }

    # Optional lifecycle hooks -------------------------------------------------

    def warmup(self) -> None:
        """Optional hook executed immediately after *load_model*."""

    def teardown(self) -> None:
        """Optional hook executed before the registry unloads the model."""

    def save_model(self, model_dir: str) -> None:
        raise NotImplementedError(f"Model saving not implemented for {self.model_name}")

    # Optional training hook ---------------------------------------------------

    def train(self, *args, **kwargs) -> None:
        """Optional hook for training a model instance.

        By default models do not support in-process training; subclasses can
        override this method if they provide an implementation.
        """

        raise NotImplementedError(f"Training is not supported for {self.model_name}")


class RecommenderException(Exception):
    """Base exception for recommender specific failures."""


class ModelNotLoadedException(RecommenderException):
    """Raised when attempting to use a model that has not been loaded."""


class UserNotFoundException(RecommenderException):
    """Raised when a user cannot be found in the training data."""


class ItemNotFoundException(RecommenderException):
    """Raised when an item cannot be found in the training data."""
