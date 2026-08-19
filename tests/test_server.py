"""Unit tests for server service layer and API request handling."""

import unittest

from smartcart.server.service import RecommendationService


class TestServerService(unittest.TestCase):
    def test_service_initialization_and_catalog(self):
        service = RecommendationService(data_dir="nonexistent_data")
        items = service.get_catalog_items(limit=10)
        self.assertEqual(len(items), 10)
        self.assertIn("name", items[0])
        self.assertIn("price", items[0])

    def test_service_recommendations(self):
        service = RecommendationService(data_dir="nonexistent_data")
        recs = service.get_recommendations(user_id=1, cart_item_ids=[0, 1], top_k=3)
        self.assertIsInstance(recs, list)
        self.assertTrue(len(recs) <= 3)


if __name__ == "__main__":
    unittest.main()
