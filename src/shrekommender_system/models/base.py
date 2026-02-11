"""Backward compatible entry point for recommender base classes."""

from shrekommender_system.recommenders.base import (  # noqa: F401
    BaseRecommender,
    ItemNotFoundException,
    ModelNotLoadedException,
    RecommendationResult,
    RecommenderException,
    SimilarItemsResult,
    UserNotFoundException,
)
