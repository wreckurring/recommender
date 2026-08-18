"""Inference pipeline for real-time Smart Cart checkout recommendations."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

from smartcart.data.catalog import ItemCatalog, Product
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.models.base import BaseRecommender


@dataclass
class CartItemRecommendation:
    item_id: int
    name: str
    category: str
    price: float
    score: float
    explanation: str


@dataclass
class CartContext:
    user_id: int
    cart_items: List[int]
    total_cart_value: float = 0.0


class SmartCartEngine:
    """End-to-end inference service for generating complementary checkout suggestions."""

    def __init__(
        self,
        model: BaseRecommender,
        catalog: ItemCatalog,
        preprocessor: InteractionPreprocessor,
        max_per_category: int = 2,
        complement_weight: float = 0.4,
    ) -> None:
        self.model = model
        self.catalog = catalog
        self.preprocessor = preprocessor
        self.max_per_category = max_per_category
        self.complement_weight = complement_weight

    def recommend_for_cart(
        self,
        user_id: int,
        cart_item_ids: List[int],
        top_k: int = 4,
        max_price_ratio: float = 1.5,
    ) -> List[CartItemRecommendation]:
        """Generate diverse, complementary recommendations for an active checkout cart."""
        # Map user ID to contiguous index
        user_idx = self.preprocessor.user_to_idx.get(user_id, -1)

        # Map cart item IDs to indices
        cart_indices = [
            self.preprocessor.item_to_idx[iid]
            for iid in cart_item_ids
            if iid in self.preprocessor.item_to_idx
        ]

        # 1. Base Model Scoring
        base_scores = self.model.score(user_idx=user_idx, cart_items=cart_indices)

        # 2. Complement Graph Boost
        # Find explicit graph complements of items currently in cart
        cart_complements: Set[int] = set()
        for iid in cart_item_ids:
            complements = self.catalog.get_complements(iid)
            cart_complements.update(complements)

        combined_scores = base_scores.copy()
        for comp_id in cart_complements:
            comp_idx = self.preprocessor.item_to_idx.get(comp_id)
            if comp_idx is not None and 0 <= comp_idx < len(combined_scores):
                # Boost candidate score
                combined_scores[comp_idx] += self.complement_weight

        # Filter out items already in the active cart
        cart_set = set(cart_item_ids)
        ranked_indices = np.argsort(-combined_scores)

        # 3. Diversity and Category Constraint Re-ranking
        selected_recs: List[CartItemRecommendation] = []
        category_counts: Dict[str, int] = {}
        total_cart_value = sum(self.catalog.get_price(i) for i in cart_item_ids)

        for idx in ranked_indices:
            raw_item_id = self.preprocessor.idx_to_item.get(int(idx))
            if raw_item_id is None or raw_item_id in cart_set:
                continue

            product = self.catalog.get_product(raw_item_id)
            if not product:
                continue

            # Check category cap
            current_cat_count = category_counts.get(product.category, 0)
            if current_cat_count >= self.max_per_category:
                continue

            # Determine human-readable explanation
            if raw_item_id in cart_complements:
                explanation = "Frequently bought together with items in your cart"
            elif user_idx >= 0:
                explanation = "Personalized for your shopping profile"
            else:
                explanation = "Popular complement in checkout"

            rec = CartItemRecommendation(
                item_id=product.item_id,
                name=product.name,
                category=product.category,
                price=product.price,
                score=float(combined_scores[idx]),
                explanation=explanation,
            )
            selected_recs.append(rec)
            category_counts[product.category] = current_cat_count + 1

            if len(selected_recs) >= top_k:
                break

        return selected_recs
