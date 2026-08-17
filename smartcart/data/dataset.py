"""PyTorch Dataset implementations and negative sampling routines for implicit CF."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class NegativeSampler:
    """Samples unobserved items per user with uniform or popularity-biased distributions."""

    def __init__(
        self,
        num_items: int,
        user_positives: Dict[int, Set[int]],
        item_frequencies: Optional[np.ndarray] = None,
        popularity_bias: float = 0.75,
        random_seed: int = 42,
    ) -> None:
        self.num_items = num_items
        self.user_positives = user_positives
        self.rng = np.random.default_rng(random_seed)

        if item_frequencies is not None:
            # Smoothed popularity distribution (Word2Vec / BPR negative sampling heuristic)
            weights = np.power(item_frequencies, popularity_bias)
            self.prob_distribution = weights / np.sum(weights)
        else:
            self.prob_distribution = None

    def sample_negative(self, user_idx: int) -> int:
        """Sample a single negative item not in user's interaction history."""
        positives = self.user_positives.get(user_idx, set())

        # If user has consumed almost all items, fallback to arbitrary unconsumed
        if len(positives) >= self.num_items:
            return 0

        while True:
            if self.prob_distribution is not None:
                neg_idx = int(self.rng.choice(self.num_items, p=self.prob_distribution))
            else:
                neg_idx = int(self.rng.integers(0, self.num_items))

            if neg_idx not in positives:
                return neg_idx


class BPRDataset(Dataset):
    """Pairwise dataset returning (user, positive_item, negative_item) triplets for BPR loss."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        num_items: int,
        user_positives: Dict[int, Set[int]],
        num_negatives: int = 1,
        item_frequencies: Optional[np.ndarray] = None,
        random_seed: int = 42,
    ) -> None:
        self.users = user_indices
        self.pos_items = item_indices
        self.num_negatives = num_negatives
        self.sampler = NegativeSampler(
            num_items=num_items,
            user_positives=user_positives,
            item_frequencies=item_frequencies,
            random_seed=random_seed,
        )

    def __len__(self) -> int:
        return len(self.users) * self.num_negatives

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_idx = idx // self.num_negatives
        user = self.users[pair_idx]
        pos_item = self.pos_items[pair_idx]
        neg_item = self.sampler.sample_negative(user)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


class PointwiseDataset(Dataset):
    """Pointwise dataset returning (user, item, binary_label) pairs for classification losses."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        num_items: int,
        user_positives: Dict[int, Set[int]],
        num_negatives: int = 4,
        random_seed: int = 42,
    ) -> None:
        self.users: List[int] = []
        self.items: List[int] = []
        self.labels: List[float] = []

        sampler = NegativeSampler(
            num_items=num_items,
            user_positives=user_positives,
            random_seed=random_seed,
        )

        for u, i in zip(user_indices, item_indices):
            # Positive sample
            self.users.append(u)
            self.items.append(i)
            self.labels.append(1.0)

            # Negative samples
            for _ in range(num_negatives):
                neg = sampler.sample_negative(u)
                self.users.append(u)
                self.items.append(neg)
                self.labels.append(0.0)

        self.users_arr = np.array(self.users, dtype=np.int64)
        self.items_arr = np.array(self.items, dtype=np.int64)
        self.labels_arr = np.array(self.labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.users_arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.users_arr[idx], dtype=torch.long),
            torch.tensor(self.items_arr[idx], dtype=torch.long),
            torch.tensor(self.labels_arr[idx], dtype=torch.float32),
        )
