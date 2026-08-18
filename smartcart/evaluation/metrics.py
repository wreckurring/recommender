"""Offline top-K ranking and catalog diversity evaluation metrics."""

from typing import Dict, List, Sequence, Set
import numpy as np


def precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Precision@K: Fraction of top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    pred_k = predicted[:k]
    actual_set = set(actual)
    if not actual_set or not pred_k:
        return 0.0
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / k)


def recall_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Recall@K: Fraction of relevant items retrieved in top-K."""
    actual_set = set(actual)
    if not actual_set or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / len(actual_set))


def f1_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute F1@K: Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(actual, predicted, k)
    r = recall_at_k(actual, predicted, k)
    if p + r == 0.0:
        return 0.0
    return float(2 * (p * r) / (p + r))


def hit_rate_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute HitRate@K: 1.0 if at least one relevant item appears in top-K, else 0.0."""
    pred_k = set(predicted[:k])
    actual_set = set(actual)
    return 1.0 if len(pred_k & actual_set) > 0 else 0.0


def ndcg_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K)."""
    pred_k = predicted[:k]
    actual_set = set(actual)
    if not actual_set or not pred_k:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(pred_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(idx + 2)  # 0-indexed: idx=0 -> rank 1 -> log2(2)

    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(k, len(actual_set))))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def mrr(actual: Sequence[int], predicted: Sequence[int]) -> float:
    """Compute Mean Reciprocal Rank (MRR) based on the rank position of the first hit."""
    actual_set = set(actual)
    for idx, item in enumerate(predicted):
        if item in actual_set:
            return float(1.0 / (idx + 1))
    return 0.0


def catalog_coverage(all_recommendations: List[List[int]], num_items: int) -> float:
    """Compute Catalog Coverage: Proportion of distinct catalog items recommended."""
    if num_items <= 0:
        return 0.0
    distinct_items: Set[int] = set()
    for rec_list in all_recommendations:
        distinct_items.update(rec_list)
    return float(len(distinct_items) / num_items)


def intra_list_diversity(
    recommendations: List[List[int]], item_to_category: Dict[int, str]
) -> float:
    """Compute Intra-List Diversity: Average distinct categories per top-K list."""
    if not recommendations:
        return 0.0
    diversities = []
    for rec_list in recommendations:
        if not rec_list:
            continue
        categories = {item_to_category.get(item, "Unknown") for item in rec_list}
        diversities.append(len(categories) / len(rec_list))
    return float(np.mean(diversities)) if diversities else 0.0
