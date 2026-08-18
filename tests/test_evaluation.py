"""Unit tests for offline evaluation metrics and benchmark harness."""

import unittest
import pandas as pd

from smartcart.evaluation.evaluator import ModelEvaluator
from smartcart.evaluation.metrics import (
    catalog_coverage,
    f1_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from smartcart.models.baselines import PopularityRecommender


class TestEvaluationMetrics(unittest.TestCase):
    def test_precision_recall_hit_rate(self):
        actual = [1, 2, 3]
        predicted = [1, 4, 5, 2, 6]

        # At k=3: predicted[:3] = [1, 4, 5], hits = 1 (item 1)
        self.assertAlmostEqual(precision_at_k(actual, predicted, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(actual, predicted, 3), 1 / 3)
        self.assertEqual(hit_rate_at_k(actual, predicted, 3), 1.0)

        # At k=1: predicted[:1] = [1], hits = 1
        self.assertAlmostEqual(precision_at_k(actual, predicted, 1), 1.0)
        self.assertAlmostEqual(recall_at_k(actual, predicted, 1), 1 / 3)

        # F1 calculation
        self.assertGreater(f1_at_k(actual, predicted, 3), 0.0)

    def test_ndcg_and_mrr(self):
        actual = [3, 7]
        predicted = [1, 3, 5, 7]

        # First hit is at index 1 (rank 2) -> MRR = 1/2 = 0.5
        self.assertAlmostEqual(mrr(actual, predicted), 0.5)

        # NDCG should be positive and <= 1.0
        ndcg_val = ndcg_at_k(actual, predicted, 4)
        self.assertGreater(ndcg_val, 0.0)
        self.assertLessEqual(ndcg_val, 1.0)

    def test_coverage_and_diversity(self):
        recs = [[0, 1], [1, 2], [3, 4]]
        cov = catalog_coverage(recs, num_items=5)
        self.assertEqual(cov, 1.0)

        item_to_cat = {0: "A", 1: "B", 2: "A", 3: "B", 4: "C"}
        div = intra_list_diversity(recs, item_to_cat)
        self.assertGreater(div, 0.0)

    def test_model_evaluator_harness(self):
        test_df = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 1],
                "item_idx": [1, 2, 0, 3],
            }
        )
        train_df = pd.DataFrame(
            {
                "user_idx": [0, 1],
                "item_idx": [0, 1],
            }
        )
        model = PopularityRecommender(num_items=4)
        model.fit(train_df)

        evaluator = ModelEvaluator(test_df=test_df, num_items=4)
        report = evaluator.evaluate_model(model, k=2)

        self.assertEqual(report.model_name, "PopularityRecommender")
        self.assertEqual(report.k, 2)
        self.assertGreaterEqual(report.precision, 0.0)


if __name__ == "__main__":
    unittest.main()
