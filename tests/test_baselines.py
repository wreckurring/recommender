"""Unit tests for baseline recommenders (Popularity, Co-occurrence)."""

import unittest
import numpy as np
import pandas as pd

from smartcart.models.baselines import ItemCooccurrenceRecommender, PopularityRecommender


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "session_id": [1, 1, 2, 2, 2, 3, 3],
                "user_idx": [0, 0, 1, 1, 1, 2, 2],
                "item_idx": [0, 1, 0, 1, 2, 2, 3],
            }
        )

    def test_popularity_recommender(self):
        model = PopularityRecommender(num_items=4)
        model.fit(self.df)

        scores = model.score(user_idx=0)
        self.assertEqual(len(scores), 4)
        # Items 0, 1, 2 have count 2; item 3 has count 1
        self.assertGreater(scores[0], scores[3])

        recs = model.recommend(user_idx=0, top_k=2, filter_items={0})
        self.assertEqual(len(recs), 2)
        self.assertNotIn(0, recs)

    def test_item_cooccurrence_recommender(self):
        model = ItemCooccurrenceRecommender(num_items=4, similarity_metric="cosine")
        model.fit(self.df)

        # Cart contains item 0, which co-occurred with item 1 twice
        recs = model.recommend(user_idx=0, top_k=2, cart_items=[0])
        self.assertNotIn(0, recs)
        self.assertIn(1, recs)


if __name__ == "__main__":
    unittest.main()
