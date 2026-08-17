"""Unit tests for PyTorch Matrix Factorization and BPR loss."""

import unittest
import torch
import numpy as np

from smartcart.models.losses import BPRLoss, PointwiseBCELoss
from smartcart.models.matrix_factorization import MatrixFactorization, MatrixFactorizationRecommender


class TestMatrixFactorization(unittest.TestCase):
    def test_forward_dimensions(self):
        num_users = 10
        num_items = 20
        dim = 8
        model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=dim)

        users = torch.tensor([0, 1, 2], dtype=torch.long)
        items = torch.tensor([5, 6, 7], dtype=torch.long)

        logits = model(users, items)
        self.assertEqual(logits.shape, (3,))

    def test_bpr_forward_and_loss(self):
        model = MatrixFactorization(num_users=10, num_items=20, embedding_dim=8)
        loss_fn = BPRLoss(l2_reg=1e-4)

        users = torch.tensor([0, 1], dtype=torch.long)
        pos = torch.tensor([2, 3], dtype=torch.long)
        neg = torch.tensor([4, 5], dtype=torch.long)

        pos_scores, neg_scores, reg_params = model.forward_bpr(users, pos, neg)
        loss = loss_fn(pos_scores, neg_scores, *reg_params)

        self.assertGreater(loss.item(), 0.0)

    def test_recommender_wrapper_scoring(self):
        rec = MatrixFactorizationRecommender(num_users=5, num_items=10, embedding_dim=4)
        scores = rec.score(user_idx=0, cart_items=[1, 2])
        self.assertEqual(len(scores), 10)

        top_recs = rec.recommend(user_idx=0, top_k=3, cart_items=[1, 2])
        self.assertEqual(len(top_recs), 3)
        self.assertNotIn(1, top_recs)
        self.assertNotIn(2, top_recs)


if __name__ == "__main__":
    unittest.main()
