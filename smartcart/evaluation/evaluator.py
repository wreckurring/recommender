"""Model evaluation harness for ranking benchmarks and comparative model analysis."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from tqdm import tqdm

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
from smartcart.models.base import BaseRecommender


@dataclass
class EvaluationReport:
    model_name: str
    k: int
    precision: float
    recall: float
    f1: float
    hit_rate: float
    ndcg: float
    mrr: float
    coverage: float
    diversity: float


class ModelEvaluator:
    """Evaluates recommendation models on held-out test sessions/interactions."""

    def __init__(
        self,
        test_df: pd.DataFrame,
        num_items: int,
        item_to_category: Optional[Dict[int, str]] = None,
        train_user_positives: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        self.test_df = test_df
        self.num_items = num_items
        self.item_to_category = item_to_category or {}
        self.train_user_positives = train_user_positives or {}

        # Group test ground-truth items by user
        self.ground_truth: Dict[int, List[int]] = {}
        for row in test_df.itertuples(index=False):
            self.ground_truth.setdefault(row.user_idx, []).append(row.item_idx)

    def evaluate_model(
        self,
        model: BaseRecommender,
        k: int = 10,
        filter_train_positives: bool = True,
        verbose: bool = False,
    ) -> EvaluationReport:
        """Run top-K ranking evaluation on all test users."""
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        hit_rates: List[float] = []
        ndcgs: List[float] = []
        mrrs: List[float] = []
        all_recs: List[List[int]] = []

        user_iter = self.ground_truth.items()
        if verbose:
            user_iter = tqdm(user_iter, desc=f"Evaluating {model.name} @ K={k}")

        for user_idx, actual_items in user_iter:
            filter_set = set(self.train_user_positives.get(user_idx, set())) if filter_train_positives else set()
            
            recs = model.recommend(
                user_idx=user_idx,
                top_k=k,
                filter_items=filter_set,
            )
            all_recs.append(recs)

            precisions.append(precision_at_k(actual_items, recs, k))
            recalls.append(recall_at_k(actual_items, recs, k))
            f1s.append(f1_at_k(actual_items, recs, k))
            hit_rates.append(hit_rate_at_k(actual_items, recs, k))
            ndcgs.append(ndcg_at_k(actual_items, recs, k))
            mrrs.append(mrr(actual_items, recs))

        cov = catalog_coverage(all_recs, self.num_items)
        div = intra_list_diversity(all_recs, self.item_to_category)

        return EvaluationReport(
            model_name=model.name,
            k=k,
            precision=float(np.mean(precisions)) if precisions else 0.0,
            recall=float(np.mean(recalls)) if recalls else 0.0,
            f1=float(np.mean(f1s)) if f1s else 0.0,
            hit_rate=float(np.mean(hit_rates)) if hit_rates else 0.0,
            ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
            mrr=float(np.mean(mrrs)) if mrrs else 0.0,
            coverage=float(cov),
            diversity=float(div),
        )

    def benchmark_models(
        self,
        models: List[BaseRecommender],
        k_values: List[int] = [5, 10, 20],
        filter_train_positives: bool = True,
    ) -> pd.DataFrame:
        """Run comprehensive benchmark across multiple models and K thresholds."""
        records = []
        for model in models:
            for k in k_values:
                report = self.evaluate_model(
                    model=model,
                    k=k,
                    filter_train_positives=filter_train_positives,
                )
                records.append(asdict(report))

        return pd.DataFrame(records)
