"""Unit tests for catalog representation and interaction data generation."""

import tempfile
import unittest
from pathlib import Path

from smartcart.config import DataConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.data.generator import InteractionGenerator


class TestDataGenerator(unittest.TestCase):
    def test_catalog_creation(self):
        catalog = ItemCatalog(num_items=100, random_seed=42)
        self.assertEqual(len(catalog.items), 100)

        prod = catalog.get_product(0)
        self.assertIsNotNone(prod)
        self.assertGreater(prod.price, 0)
        self.assertIsInstance(prod.complements, list)
        self.assertGreater(len(prod.complements), 0)

    def test_interaction_generator_generation_and_splits(self):
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            self.skipTest("pandas or numpy not installed in current environment")

        data_cfg = DataConfig(
            num_users=50,
            num_items=100,
            num_interactions=1000,
            test_ratio=0.2,
            val_ratio=0.1,
            random_seed=42,
        )
        catalog = ItemCatalog(num_items=100, random_seed=42)
        generator = InteractionGenerator(catalog, data_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_df, val_df, test_df = generator.generate_and_save(tmpdir)

            self.assertEqual(len(train_df) + len(val_df) + len(test_df), 1000)
            self.assertEqual(len(test_df), 200)
            self.assertEqual(len(val_df), 100)
            self.assertEqual(len(train_df), 700)

            self.assertTrue(Path(tmpdir, "catalog.csv").exists())
            self.assertTrue(Path(tmpdir, "interactions.csv").exists())
            self.assertTrue(Path(tmpdir, "train.csv").exists())
            self.assertTrue(Path(tmpdir, "val.csv").exists())
            self.assertTrue(Path(tmpdir, "test.csv").exists())

            self.assertIn("session_id", train_df.columns)
            self.assertIn("user_id", train_df.columns)
            self.assertIn("item_id", train_df.columns)


if __name__ == "__main__":
    unittest.main()
