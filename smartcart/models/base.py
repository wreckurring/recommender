"""Abstract base class for all recommendation models."""

from abc import ABC, abstractmethod
from typing import List, Optional, Set
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Base interface for collaborative filtering and cart recommendation models."""

    def __init__(self, name: str = "BaseRecommender") -> None:
        self.name = name
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame) -> "BaseRecommender":
        """Train or build index from interaction logs."""
        pass

    @abstractmethod
    def score(self, user_idx: int, item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict scores for given items (or all items if item_indices is None)."""
        pass

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_items: Optional[Set[int]] = None,
        cart_items: Optional[List[int]] = None,
    ) -> List[int]:
        """Generate top-K item recommendations, excluding filtered items."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating recommendations.")

        scores = self.score(user_idx=user_idx)
        if filter_items:
            for item in filter_items:
                if 0 <= item < len(scores):
                    scores[item] = -np.inf

        # Rank items descending
        ranked_indices = np.argsort(-scores)
        top_items = [int(idx) for idx in ranked_indices if scores[idx] > -np.inf][:top_k]
        return top_items
