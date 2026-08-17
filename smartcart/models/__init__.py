"""Recommendation models: Baselines, Matrix Factorization, and Cart-aware architectures."""

from smartcart.models.base import BaseRecommender
from smartcart.models.baselines import ItemCooccurrenceRecommender, PopularityRecommender

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "ItemCooccurrenceRecommender",
]
