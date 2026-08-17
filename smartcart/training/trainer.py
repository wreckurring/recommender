"""Training loop, early stopping, and checkpoint management for recommendation models."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smartcart.config import ModelConfig
from smartcart.data.dataset import BPRDataset
from smartcart.models.losses import BPRLoss
from smartcart.models.matrix_factorization import MatrixFactorization, MatrixFactorizationRecommender


class EarlyStopping:
    """Monitors validation loss and triggers early termination when progress stalls."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelTrainer:
    """Encapsulates training, validation, and serialization of Matrix Factorization models."""

    def __init__(
        self,
        model: MatrixFactorization,
        config: ModelConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.loss_fn = BPRLoss(l2_reg=config.weight_decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.early_stopping = EarlyStopping(patience=3)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for users, pos_items, neg_items in dataloader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            self.optimizer.zero_grad()
            pos_scores, neg_scores, reg_params = self.model.forward_bpr(users, pos_items, neg_items)
            loss = self.loss_fn(pos_scores, neg_scores, *reg_params)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate_loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        for users, pos_items, neg_items in dataloader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            pos_scores, neg_scores, reg_params = self.model.forward_bpr(users, pos_items, neg_items)
            loss = self.loss_fn(pos_scores, neg_scores, *reg_params)
            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Run full training routine over configured epochs with early stopping."""
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate_loss(val_loader)
                self.history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

                if verbose:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch:02d}/{self.config.num_epochs:02d} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}"
                    )

                if self.early_stopping.step(val_loss):
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch:02d}/{self.config.num_epochs:02d} | Train Loss: {train_loss:.4f}")

        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state dictionary and hyperparameters."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_users": self.model.num_users,
                "num_items": self.model.num_items,
                "embedding_dim": self.model.embedding_dim,
                "history": self.history,
            },
            save_path,
        )

    @classmethod
    def load_recommender(
        cls, checkpoint_path: str | Path, device: Optional[torch.device] = None
    ) -> MatrixFactorizationRecommender:
        """Instantiate a fitted MatrixFactorizationRecommender from checkpoint."""
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=target_device)

        recommender = MatrixFactorizationRecommender(
            num_users=ckpt["num_users"],
            num_items=ckpt["num_items"],
            embedding_dim=ckpt["embedding_dim"],
            device=str(target_device),
        )
        recommender.model.load_state_dict(ckpt["model_state_dict"])
        recommender.is_fitted = True
        return recommender
