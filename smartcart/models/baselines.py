"""Baseline recommendation models: Popularity and Item Co-occurrence CF."""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
import scipy.sparse as sp

from smartcart.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends globally most popular items across historical checkout sessions."""

    def __init__(self, num_items: int) -> None:
        super().__init__(name="PopularityRecommender")
        self.num_items = num_items
        self.item_scores = np.zeros(num_items, dtype=np.float32)

    def fit(self, interactions_df: pd.DataFrame) -> "PopularityRecommender":
        counts = interactions_df["item_idx"].value_counts()
        for item_idx, count in counts.items():
            if item_idx < self.num_items:
                self.item_scores[item_idx] = float(count)
        
        # Normalize scores to [0, 1]
        max_val = self.item_scores.max()
        if max_val > 0:
            self.item_scores /= max_val

        self.is_fitted = True
        return self

    def score(self, user_idx: int, item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        if item_indices is not None:
            return self.item_scores[item_indices].copy()
        return self.item_scores.copy()


class ItemCooccurrenceRecommender(BaseRecommender):
    """Recommends items based on session co-occurrence similarity matrix (Jaccard / Cosine)."""

    def __init__(self, num_items: int, similarity_metric: str = "cosine") -> None:
        super().__init__(name="ItemCooccurrenceRecommender")
        self.num_items = num_items
        self.similarity_metric = similarity_metric
        self.similarity_matrix = np.zeros((num_items, num_items), dtype=np.float32)

    def fit(self, interactions_df: pd.DataFrame) -> "ItemCooccurrenceRecommender":
        # Group by session to construct session-item binary matrix
        sessions = interactions_df["session_id"].unique()
        session_to_idx = {sid: idx for idx, sid in enumerate(sessions)}

        row_indices = interactions_df["session_id"].map(session_to_idx).values
        col_indices = interactions_df["item_idx"].values
        data = np.ones(len(interactions_df), dtype=np.float32)

        session_item_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(sessions), self.num_items),
        )
        # Binarize
        session_item_matrix.data = np.ones_like(session_item_matrix.data)

        # Item-item co-occurrence: C = X^T * X
        cooccurrence = (session_item_matrix.T @ session_item_matrix).toarray()
        np.fill_diagonal(cooccurrence, 0.0)

        if self.similarity_metric == "cosine":
            item_counts = session_item_matrix.sum(axis=0).A1
            norms = np.sqrt(item_counts[:, None] * item_counts[None, :])
            norms[norms == 0] = 1.0
            self.similarity_matrix = (cooccurrence / norms).astype(np.float32)
        elif self.similarity_metric == "jaccard":
            item_counts = session_item_matrix.sum(axis=0).A1
            union = item_counts[:, None] + item_counts[None, :] - cooccurrence
            union[union == 0] = 1.0
            self.similarity_matrix = (cooccurrence / union).astype(np.float32)
        else:
            self.similarity_matrix = cooccurrence.astype(np.float32)

        self.is_fitted = True
        return self

    def score(
        self,
        user_idx: int,
        item_indices: Optional[np.ndarray] = None,
        cart_items: Optional[List[int]] = None,
    ) -> np.ndarray:
        if cart_items is None or len(cart_items) == 0:
            # Fallback to mean item similarity across catalog
            scores = self.similarity_matrix.mean(axis=0)
        else:
            valid_cart = [i for i in cart_items if 0 <= i < self.num_items]
            if not valid_cart:
                scores = self.similarity_matrix.mean(axis=0)
            else:
                # Sum affinity vectors of all items in current cart
                scores = self.similarity_matrix[valid_cart].sum(axis=0)

        if item_indices is not None:
            return scores[item_indices].copy()
        return scores.copy()

    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        filter_items: Optional[Set[int]] = None,
        cart_items: Optional[List[int]] = None,
    ) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before generating recommendations.")

        scores = self.score(user_idx=user_idx, cart_items=cart_items)

        # Exclude items currently in cart as well as filter_items
        exclude = set(filter_items or set())
        if cart_items:
            exclude.update(cart_items)

        for item in exclude:
            if 0 <= item < len(scores):
                scores[item] = -np.inf

        ranked_indices = np.argsort(-scores)
        top_items = [int(idx) for idx in ranked_indices if scores[idx] > -np.inf][:top_k]
        return top_items
