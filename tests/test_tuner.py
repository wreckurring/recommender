"""Unit tests for HyperparameterTuner grid and random search sweeps."""

import unittest
import pandas as pd
import torch

from smartcart.training.tuner import HyperparameterTuner, TrialResult


class TestTuner(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 1, 2, 2],
                "item_idx": [0, 1, 1, 2, 2, 3],
            }
        )
        self.val_df = pd.DataFrame(
            {
                "user_idx": [0, 1, 2],
                "item_idx": [2, 0, 1],
            }
        )
        self.user_positives = {0: {0, 1}, 1: {1, 2}, 2: {2, 3}}

    def test_random_search_sweep(self):
        tuner = HyperparameterTuner(
            train_df=self.train_df,
            val_df=self.val_df,
            num_users=3,
            num_items=4,
            user_positives=self.user_positives,
            device=torch.device("cpu"),
        )
        distributions = {
            "embedding_dim": [4, 8],
            "learning_rate": [0.01, 0.005],
            "weight_decay": [1e-4],
            "num_negatives": [1],
            "batch_size": [4],
        }

        trials = tuner.run_random_search(
            param_distributions=distributions,
            num_trials=2,
            k=2,
            epochs_per_trial=1,
            verbose=False,
        )

        self.assertEqual(len(trials), 2)
        best = tuner.get_best_trial(trials, metric="f1_at_k")
        self.assertIsInstance(best, TrialResult)
        self.assertGreaterEqual(best.f1_at_k, 0.0)


if __name__ == "__main__":
    unittest.main()
