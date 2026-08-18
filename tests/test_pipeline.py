"""Unit tests for SmartCartEngine inference and diversity re-ranking."""

import unittest
import pandas as pd

from smartcart.data.catalog import ItemCatalog
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.models.baselines import PopularityRecommender
from smartcart.pipeline.engine import SmartCartEngine


class TestSmartCartEngine(unittest.TestCase):
    def setUp(self):
        self.catalog = ItemCatalog(num_items=20, random_seed=42)
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "item_id": [0, 1, 1, 2, 3],
                "timestamp": [10, 20, 30, 40, 50],
            }
        )
        self.preprocessor = InteractionPreprocessor()
        transformed = self.preprocessor.fit_transform(df)

        self.model = PopularityRecommender(num_items=len(self.preprocessor.item_to_idx))
        self.model.fit(transformed)

    def test_recommend_excludes_cart_items(self):
        engine = SmartCartEngine(
            model=self.model,
            catalog=self.catalog,
            preprocessor=self.preprocessor,
            max_per_category=2,
        )

        cart_items = [0]
        recs = engine.recommend_for_cart(user_id=1, cart_item_ids=cart_items, top_k=3)

        self.assertGreater(len(recs), 0)
        rec_ids = [r.item_id for r in recs]
        self.assertNotIn(0, rec_ids)
        self.assertTrue(all(r.price > 0 for r in recs))
        self.assertTrue(all(r.explanation for r in recs))


if __name__ == "__main__":
    unittest.main()
