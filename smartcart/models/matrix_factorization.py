"""PyTorch Matrix Factorization and Cart-Aware Collaborative Filtering architectures."""

from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from smartcart.models.base import BaseRecommender


class MatrixFactorization(nn.Module):
    """PyTorch Matrix Factorization with latent user/item embeddings and bias terms."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Latent factor embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Weight initialization (normal distribution with low variance)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Pointwise dot-product scoring for batches of (user, item) pairs."""
        u_emb = self.user_embedding(user_indices)  # [B, d]
        i_emb = self.item_embedding(item_indices)  # [B, d]
        u_b = self.user_bias(user_indices).squeeze(-1)  # [B]
        i_b = self.item_bias(item_indices).squeeze(-1)  # [B]

        interaction = (u_emb * i_emb).sum(dim=-1)
        logits = self.global_bias + u_b + i_b + interaction
        return logits

    def forward_bpr(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Compute positive and negative predicted scores for pairwise BPR optimization."""
        u_emb = self.user_embedding(users)
        pos_i_emb = self.item_embedding(pos_items)
        neg_i_emb = self.item_embedding(neg_items)

        u_b = self.user_bias(users).squeeze(-1)
        pos_i_b = self.item_bias(pos_items).squeeze(-1)
        neg_i_b = self.item_bias(neg_items).squeeze(-1)

        pos_scores = self.global_bias + u_b + pos_i_b + (u_emb * pos_i_emb).sum(dim=-1)
        neg_scores = self.global_bias + u_b + neg_i_b + (u_emb * neg_i_emb).sum(dim=-1)

        regularized_params = [u_emb, pos_i_emb, neg_i_emb, u_b, pos_i_b, neg_i_b]
        return pos_scores, neg_scores, regularized_params

    @torch.no_grad()
    def get_all_item_scores(self, user_idx: int, device: torch.device) -> np.ndarray:
        """Efficient vectorized scoring for all catalog items for a single user."""
        self.eval()
        u_idx = torch.tensor([user_idx], dtype=torch.long, device=device)
        u_emb = self.user_embedding(u_idx)  # [1, d]
        u_b = self.user_bias(u_idx).item()  # float

        all_item_emb = self.item_embedding.weight  # [I, d]
        all_item_b = self.item_bias.weight.squeeze(-1)  # [I]

        # Dot product across full item matrix
        scores = self.global_bias.item() + u_b + all_item_b + torch.matmul(all_item_emb, u_emb.squeeze(0))
        return scores.detach().cpu().numpy()


class MatrixFactorizationRecommender(BaseRecommender):
    """Wrapper implementing BaseRecommender around PyTorch MatrixFactorization model."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name="MatrixFactorization")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MatrixFactorization(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        ).to(self.device)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

    def fit(self, interactions_df: pd.DataFrame) -> "MatrixFactorizationRecommender":
        """Note: Detailed training loop is handled by smartcart.training.Trainer."""
        self.is_fitted = True
        return self

    def score(
        self,
        user_idx: int,
        item_indices: Optional[np.ndarray] = None,
        cart_items: Optional[List[int]] = None,
    ) -> np.ndarray:
        if not (0 <= user_idx < self.num_users):
            # Cold-start user fallback: mean item bias
            all_scores = (
                self.model.global_bias.item()
                + self.model.item_bias.weight.squeeze(-1).detach().cpu().numpy()
            )
        else:
            all_scores = self.model.get_all_item_scores(user_idx, self.device)

        # Contextual Cart Complement Boost: If active cart exists, compute item-item latent affinity
        if cart_items:
            valid_cart = [c for c in cart_items if 0 <= c < self.num_items]
            if valid_cart:
                cart_t = torch.tensor(valid_cart, dtype=torch.long, device=self.device)
                cart_embs = self.model.item_embedding(cart_t)  # [C, d]
                avg_cart_emb = cart_embs.mean(dim=0)  # [d]

                all_item_emb = self.model.item_embedding.weight  # [I, d]
                cart_affinity = torch.matmul(all_item_emb, avg_cart_emb).detach().cpu().numpy()
                all_scores += 0.5 * cart_affinity

        if item_indices is not None:
            return all_scores[item_indices]
        return all_scores
