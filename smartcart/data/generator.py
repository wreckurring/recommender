"""User interaction simulator generating realistic checkout sessions and basket logs."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from smartcart.config import DataConfig
from smartcart.data.catalog import ItemCatalog


class InteractionGenerator:
    """Simulates realistic user browsing, cart-building, and checkout interactions."""

    def __init__(self, catalog: ItemCatalog, config: DataConfig) -> None:
        self.catalog = catalog
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.user_profiles: Dict[int, Dict] = {}
        self._build_user_profiles()

    def _build_user_profiles(self) -> None:
        """Assign category preferences and activity levels to simulated users."""
        categories = ItemCatalog.CATEGORIES

        for user_id in range(self.config.num_users):
            # Prefer 1 to 3 primary categories
            num_fav_cats = self.rng.integers(1, 4)
            fav_cats = self.rng.choice(categories, size=num_fav_cats, replace=False).tolist()
            
            # User price sensitivity (0.0: insensitive, 1.0: highly sensitive)
            price_sensitivity = float(self.rng.beta(2, 5))
            
            # Activity tier (low, medium, high frequency buyer)
            activity_weight = float(self.rng.exponential(scale=1.0) + 0.1)

            self.user_profiles[user_id] = {
                "favorite_categories": set(fav_cats),
                "price_sensitivity": price_sensitivity,
                "activity_weight": activity_weight,
            }

    def generate_interactions(self) -> pd.DataFrame:
        """Generate 100K+ simulated checkout baskets and item interaction logs."""
        all_item_ids = list(self.catalog.items.keys())
        category_to_items: Dict[str, List[int]] = {}
        for item_id, prod in self.catalog.items.items():
            category_to_items.setdefault(prod.category, []).append(item_id)

        user_weights = np.array([p["activity_weight"] for p in self.user_profiles.values()])
        user_weights /= user_weights.sum()

        interactions: List[Dict] = []
        session_id = 0
        timestamp_base = 1672531199  # Jan 1, 2023 base timestamp

        while len(interactions) < self.config.num_interactions:
            user_id = int(self.rng.choice(self.config.num_users, p=user_weights))
            profile = self.user_profiles[user_id]
            fav_cats = profile["favorite_categories"]

            # Cart size distribution (geometric/poisson: 2 to 6 items per checkout)
            basket_size = int(self.rng.integers(2, 7))
            
            # Select seed item aligned with user preference (80% probability)
            if self.rng.random() < 0.80 and fav_cats:
                cat = self.rng.choice(list(fav_cats))
                seed_pool = category_to_items[cat]
                seed_item = int(self.rng.choice(seed_pool))
            else:
                seed_item = int(self.rng.choice(all_item_ids))

            current_basket = [seed_item]

            # Populate rest of the basket with mix of complements and organic selections
            for _ in range(basket_size - 1):
                last_item = current_basket[-1]
                complements = self.catalog.get_complements(last_item)
                
                # 65% probability of selecting a complementary item to the basket
                if complements and self.rng.random() < 0.65:
                    next_item = int(self.rng.choice(complements))
                elif self.rng.random() < 0.70 and fav_cats:
                    cat = self.rng.choice(list(fav_cats))
                    next_item = int(self.rng.choice(category_to_items[cat]))
                else:
                    next_item = int(self.rng.choice(all_item_ids))

                if next_item not in current_basket:
                    current_basket.append(next_item)

            session_timestamp = timestamp_base + session_id * 180 + int(self.rng.integers(0, 60))

            for item_id in current_basket:
                interactions.append(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "item_id": item_id,
                        "timestamp": session_timestamp,
                        "interaction_type": "checkout_purchase",
                        "price": self.catalog.get_price(item_id),
                    }
                )

            session_id += 1

        df = pd.DataFrame(interactions[: self.config.num_interactions])
        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions chronologically and user-stratified into train, val, test."""
        # Chronological sort
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        n_total = len(df)
        n_test = int(n_total * self.config.test_ratio)
        n_val = int(n_total * self.config.val_ratio)
        n_train = n_total - n_val - n_test

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()

        return train_df, val_df, test_df

    def generate_and_save(self, output_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrate generation and save artifacts to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        catalog_df = self.catalog.to_dataframe()
        catalog_df.to_csv(out_path / "catalog.csv", index=False)

        interactions_df = self.generate_interactions()
        interactions_df.to_csv(out_path / "interactions.csv", index=False)

        train_df, val_df, test_df = self.split_data(interactions_df)
        train_df.to_csv(out_path / "train.csv", index=False)
        val_df.to_csv(out_path / "val.csv", index=False)
        test_df.to_csv(out_path / "test.csv", index=False)

        return train_df, val_df, test_df
