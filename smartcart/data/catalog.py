"""Product catalog representation and complementary product graph generation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd


@dataclass
class Product:
    item_id: int
    name: str
    category: str
    price: float
    complements: List[int] = field(default_factory=list)


class ItemCatalog:
    """Manages the product catalog and domain-aware complement relationships."""

    CATEGORIES = [
        "Electronics",
        "Home & Kitchen",
        "Groceries",
        "Sports & Fitness",
        "Apparel",
        "Beauty & Grooming",
        "Books & Stationery",
        "Health & Wellness",
    ]

    CATEGORY_COMPLEMENT_MAP = {
        "Electronics": ["Electronics", "Home & Kitchen"],
        "Home & Kitchen": ["Groceries", "Home & Kitchen"],
        "Groceries": ["Groceries", "Home & Kitchen"],
        "Sports & Fitness": ["Health & Wellness", "Sports & Fitness", "Apparel"],
        "Apparel": ["Apparel", "Beauty & Grooming"],
        "Beauty & Grooming": ["Beauty & Grooming", "Health & Wellness"],
        "Books & Stationery": ["Books & Stationery", "Electronics"],
        "Health & Wellness": ["Groceries", "Health & Wellness"],
    }

    CATEGORY_PRICE_RANGES = {
        "Electronics": (25.0, 450.0),
        "Home & Kitchen": (15.0, 120.0),
        "Groceries": (3.0, 35.0),
        "Sports & Fitness": (18.0, 180.0),
        "Apparel": (15.0, 95.0),
        "Beauty & Grooming": (8.0, 60.0),
        "Books & Stationery": (6.0, 40.0),
        "Health & Wellness": (10.0, 75.0),
    }

    def __init__(self, num_items: int = 500, random_seed: int = 42) -> None:
        self.num_items = num_items
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.items: Dict[int, Product] = {}
        self._build_catalog()

    def _build_catalog(self) -> None:
        category_item_buckets: Dict[str, List[int]] = {cat: [] for cat in self.CATEGORIES}

        # Step 1: Instantiate products with realistic price distributions
        for item_id in range(self.num_items):
            category = self.CATEGORIES[item_id % len(self.CATEGORIES)]
            min_p, max_p = self.CATEGORY_PRICE_RANGES[category]
            price = float(np.round(self.rng.uniform(min_p, max_p), 2))
            name = f"{category.split()[0]}_Item_{item_id:04d}"

            self.items[item_id] = Product(
                item_id=item_id,
                name=name,
                category=category,
                price=price,
                complements=[],
            )
            category_item_buckets[category].append(item_id)

        # Step 2: Establish complementary product graph edges
        for item_id, product in self.items.items():
            valid_complement_cats = self.CATEGORY_COMPLEMENT_MAP.get(product.category, [product.category])
            candidate_pool = [
                cid
                for cat in valid_complement_cats
                for cid in category_item_buckets[cat]
                if cid != item_id
            ]

            num_complements = self.rng.integers(3, 8)
            if candidate_pool:
                chosen = self.rng.choice(
                    candidate_pool,
                    size=min(num_complements, len(candidate_pool)),
                    replace=False,
                ).tolist()
                product.complements = chosen

    def get_product(self, item_id: int) -> Optional[Product]:
        return self.items.get(item_id)

    def get_price(self, item_id: int) -> float:
        prod = self.items.get(item_id)
        return prod.price if prod else 0.0

    def get_complements(self, item_id: int) -> List[int]:
        prod = self.items.get(item_id)
        return prod.complements if prod else []

    def to_dataframe(self) -> pd.DataFrame:
        records = [
            {
                "item_id": p.item_id,
                "name": p.name,
                "category": p.category,
                "price": p.price,
                "complements": ",".join(map(str, p.complements)),
            }
            for p in self.items.values()
        ]
        return pd.DataFrame(records)
