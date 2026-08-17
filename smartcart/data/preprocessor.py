"""Data preprocessing, ID mapping, and interaction matrix generation."""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class DatasetMetadata:
    num_users: int
    num_items: int
    num_interactions: int
    density: float


class InteractionPreprocessor:
    """Encodes raw user/item IDs into contiguous indices and constructs interaction matrices."""

    def __init__(self) -> None:
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "InteractionPreprocessor":
        """Fit user and item ID vocabularies from interaction logs."""
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}

        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for idx, iid in enumerate(unique_items)}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw IDs to continuous indices, filtering unknown entities."""
        if not self.is_fitted:
            raise RuntimeError("InteractionPreprocessor must be fitted before calling transform.")

        df_out = df.copy()
        df_out = df_out[
            df_out["user_id"].isin(self.user_to_idx) & df_out["item_id"].isin(self.item_to_idx)
        ].copy()

        df_out["user_idx"] = df_out["user_id"].map(self.user_to_idx)
        df_out["item_idx"] = df_out["item_id"].map(self.item_to_idx)
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_metadata(self, df: pd.DataFrame) -> DatasetMetadata:
        """Compute summary statistics and matrix density."""
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        n_interactions = len(df)
        density = n_interactions / (n_users * n_items) if n_users and n_items else 0.0

        return DatasetMetadata(
            num_users=n_users,
            num_items=n_items,
            num_interactions=n_interactions,
            density=float(density),
        )

    def build_interaction_matrix(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Construct sparse CSR matrix of user-item interactions."""
        users = df["user_idx"].values
        items = df["item_idx"].values
        data = np.ones(len(df), dtype=np.float32)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        return sp.csr_matrix((data, (users, items)), shape=(n_users, n_items))

    def get_user_positive_items(self, df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Map each user index to a set of interacted item indices for negative filtering."""
        user_pos: Dict[int, Set[int]] = {}
        for row in df[["user_idx", "item_idx"]].itertuples(index=False):
            user_pos.setdefault(row.user_idx, set()).add(row.item_idx)
        return user_pos
