"""Recommendation models: Baselines, Matrix Factorization, and Cart-aware architectures."""

from smartcart.models.base import BaseRecommender
from smartcart.models.baselines import ItemCooccurrenceRecommender, PopularityRecommender
from smartcart.models.losses import BPRLoss, PointwiseBCELoss
from smartcart.models.matrix_factorization import (
    MatrixFactorization,
    MatrixFactorizationRecommender,
)

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "ItemCooccurrenceRecommender",
    "MatrixFactorization",
    "MatrixFactorizationRecommender",
    "BPRLoss",
    "PointwiseBCELoss",
]
