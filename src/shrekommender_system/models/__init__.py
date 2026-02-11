"""
Models package for the shrekommender system

This package contains all recommendation models and the inference engine.
"""

from .base import BaseRecommender, RecommendationResult, SimilarItemsResult
from .als_recommender import ALSRecommender

__all__ = [
    'BaseRecommender',
    'RecommendationResult', 
    'SimilarItemsResult',
    'ALSRecommender',
]
