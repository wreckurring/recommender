"""Offline evaluation metrics and benchmark harness for recommendation models."""

from smartcart.evaluation.evaluator import EvaluationReport, ModelEvaluator
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

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "mrr",
    "catalog_coverage",
    "intra_list_diversity",
    "EvaluationReport",
    "ModelEvaluator",
]
