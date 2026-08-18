"""Hyperparameter optimization and grid/random search routines for recommendation models."""

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from smartcart.config import ModelConfig
from smartcart.data.dataset import BPRDataset
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.evaluation.evaluator import ModelEvaluator
from smartcart.models.matrix_factorization import MatrixFactorization, MatrixFactorizationRecommender
from smartcart.training.trainer import ModelTrainer


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    ndcg_at_k: float
    val_loss: float


class HyperparameterTuner:
    """Explores hyperparameter configurations to optimize ranking metrics and precision/recall trade-offs."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_positives: Dict[int, set],
        device: Optional[torch.device] = None,
        random_seed: int = 42,
    ) -> None:
        self.train_df = train_df
        self.val_df = val_df
        self.num_users = num_users
        self.num_items = num_items
        self.user_positives = user_positives
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(random_seed)

        self.evaluator = ModelEvaluator(
            test_df=val_df,
            num_items=num_items,
            train_user_positives=user_positives,
        )

    def run_grid_search(
        self,
        param_grid: Dict[str, Sequence[Any]],
        k: int = 10,
        epochs_per_trial: int = 5,
        verbose: bool = True,
    ) -> List[TrialResult]:
        """Execute exhaustive grid search across parameter space."""
        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))

        trials: List[TrialResult] = []
        if verbose:
            print(f"Starting grid search: {len(combinations)} total configurations.")

        for idx, values in enumerate(combinations):
            params = dict(zip(keys, values))
            trial = self._evaluate_configuration(
                trial_id=idx + 1,
                params=params,
                k=k,
                epochs=epochs_per_trial,
                verbose=verbose,
            )
            trials.append(trial)

        return trials

    def run_random_search(
        self,
        param_distributions: Dict[str, Sequence[Any]],
        num_trials: int = 8,
        k: int = 10,
        epochs_per_trial: int = 5,
        verbose: bool = True,
    ) -> List[TrialResult]:
        """Sample random parameter combinations from distribution spaces."""
        trials: List[TrialResult] = []
        if verbose:
            print(f"Starting random search: sampling {num_trials} configurations.")

        for idx in range(num_trials):
            params = {
                k: self.rng.choice(v).item() if isinstance(v, np.ndarray) else self.rng.choice(v)
                for k, v in param_distributions.items()
            }
            trial = self._evaluate_configuration(
                trial_id=idx + 1,
                params=params,
                k=k,
                epochs=epochs_per_trial,
                verbose=verbose,
            )
            trials.append(trial)

        return trials

    def _evaluate_configuration(
        self,
        trial_id: int,
        params: Dict[str, Any],
        k: int,
        epochs: int,
        verbose: bool = True,
    ) -> TrialResult:
        """Train candidate model and compute validation ranking metrics."""
        cfg = ModelConfig(
            embedding_dim=params.get("embedding_dim", 32),
            learning_rate=params.get("learning_rate", 0.001),
            weight_decay=params.get("weight_decay", 1e-4),
            batch_size=params.get("batch_size", 512),
            num_epochs=epochs,
        )

        train_dataset = BPRDataset(
            user_indices=self.train_df["user_idx"].values,
            item_indices=self.train_df["item_idx"].values,
            num_items=self.num_items,
            user_positives=self.user_positives,
            num_negatives=params.get("num_negatives", 2),
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

        model = MatrixFactorization(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=cfg.embedding_dim,
        ).to(self.device)

        trainer = ModelTrainer(model=model, config=cfg, device=self.device)
        history = trainer.fit(train_loader=train_loader, verbose=False)
        final_val_loss = history["train_loss"][-1] if history["train_loss"] else 0.0

        # Wrap in recommender for offline evaluation
        recommender = MatrixFactorizationRecommender(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=cfg.embedding_dim,
            device=str(self.device),
        )
        recommender.model = model
        recommender.is_fitted = True

        report = self.evaluator.evaluate_model(recommender, k=k)

        if verbose:
            print(
                f"Trial {trial_id:02d} | Dim: {cfg.embedding_dim:2d}, LR: {cfg.learning_rate:.4f}, "
                f"WD: {cfg.weight_decay:.1e} -> P@{k}: {report.precision:.4f}, "
                f"R@{k}: {report.recall:.4f}, F1@{k}: {report.f1:.4f}, NDCG@{k}: {report.ndcg:.4f}"
            )

        return TrialResult(
            trial_id=trial_id,
            params=params,
            precision_at_k=report.precision,
            recall_at_k=report.recall,
            f1_at_k=report.f1,
            ndcg_at_k=report.ndcg,
            val_loss=final_val_loss,
        )

    @staticmethod
    def get_best_trial(trials: List[TrialResult], metric: str = "f1_at_k") -> TrialResult:
        """Retrieve trial maximizing the specified target metric."""
        return max(trials, key=lambda t: getattr(t, metric))
