"""Unit tests for ModelTrainer, early stopping, and checkpoint serialization."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from smartcart.config import ModelConfig
from smartcart.data.dataset import BPRDataset
from smartcart.models.matrix_factorization import MatrixFactorization
from smartcart.training.trainer import EarlyStopping, ModelTrainer


class TestTrainer(unittest.TestCase):
    def test_early_stopping(self):
        es = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(es.step(1.0))
        self.assertFalse(es.step(0.95))
        self.assertFalse(es.step(0.96))
        self.assertTrue(es.step(0.97))  # Counter reaches patience=2

    def test_trainer_fit_and_checkpoint(self):
        cfg = ModelConfig(
            embedding_dim=4,
            learning_rate=0.01,
            weight_decay=1e-5,
            num_epochs=3,
            batch_size=8,
        )
        model = MatrixFactorization(num_users=5, num_items=10, embedding_dim=4)

        users = np.array([0, 1, 2, 3, 4])
        items = np.array([1, 2, 3, 4, 5])
        user_positives = {i: {items[i]} for i in range(5)}

        dataset = BPRDataset(
            user_indices=users,
            item_indices=items,
            num_items=10,
            user_positives=user_positives,
            num_negatives=1,
            random_seed=42,
        )
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        trainer = ModelTrainer(model=model, config=cfg, device=torch.device("cpu"))
        history = trainer.fit(train_loader=loader, val_loader=loader, verbose=False)

        self.assertEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["val_loss"]), 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            trainer.save_checkpoint(ckpt_path)
            self.assertTrue(ckpt_path.exists())

            loaded_rec = ModelTrainer.load_recommender(ckpt_path, device=torch.device("cpu"))
            self.assertTrue(loaded_rec.is_fitted)
            scores = loaded_rec.score(user_idx=0)
            self.assertEqual(len(scores), 10)


if __name__ == "__main__":
    unittest.main()
