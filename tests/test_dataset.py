"""Unit tests for dataset preprocessing and negative sampling."""

import unittest
import numpy as np
import pandas as pd
import torch

from smartcart.data.dataset import BPRDataset, NegativeSampler, PointwiseDataset
from smartcart.data.preprocessor import InteractionPreprocessor


class TestDatasetAndPreprocessor(unittest.TestCase):
    def setUp(self):
        self.raw_df = pd.DataFrame(
            {
                "user_id": [101, 101, 102, 103, 103, 103],
                "item_id": [501, 502, 501, 503, 504, 505],
                "timestamp": [1, 2, 3, 4, 5, 6],
            }
        )

    def test_preprocessor_fitting_and_transformation(self):
        prep = InteractionPreprocessor()
        transformed = prep.fit_transform(self.raw_df)

        self.assertEqual(len(prep.user_to_idx), 3)
        self.assertEqual(len(prep.item_to_idx), 5)
        self.assertIn("user_idx", transformed.columns)
        self.assertIn("item_idx", transformed.columns)

        meta = prep.get_metadata(transformed)
        self.assertEqual(meta.num_users, 3)
        self.assertEqual(meta.num_items, 5)
        self.assertEqual(meta.num_interactions, 6)

    def test_negative_sampler_excludes_positives(self):
        user_positives = {0: {1, 2, 3}, 1: {0}}
        sampler = NegativeSampler(num_items=5, user_positives=user_positives, random_seed=42)

        for _ in range(20):
            neg_0 = sampler.sample_negative(0)
            self.assertNotIn(neg_0, {1, 2, 3})
            self.assertIn(neg_0, {0, 4})

            neg_1 = sampler.sample_negative(1)
            self.assertNotIn(neg_1, {0})
            self.assertIn(neg_1, {1, 2, 3, 4})

    def test_bpr_dataset_item_types(self):
        users = np.array([0, 0, 1])
        items = np.array([1, 2, 0])
        user_positives = {0: {1, 2}, 1: {0}}

        ds = BPRDataset(
            user_indices=users,
            item_indices=items,
            num_items=5,
            user_positives=user_positives,
            num_negatives=2,
            random_seed=42,
        )

        self.assertEqual(len(ds), 6)
        u, pos_i, neg_i = ds[0]
        self.assertIsInstance(u, torch.Tensor)
        self.assertIsInstance(pos_i, torch.Tensor)
        self.assertIsInstance(neg_i, torch.Tensor)
        self.assertEqual(u.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
