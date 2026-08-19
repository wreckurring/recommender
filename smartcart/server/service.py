"""Service layer managing models, catalog, and A/B simulation state for API requests."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from smartcart.ab_testing.simulator import ABExperimentSimulator
from smartcart.config import ABTestConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.models.baselines import ItemCooccurrenceRecommender, PopularityRecommender
from smartcart.models.matrix_factorization import MatrixFactorizationRecommender
from smartcart.pipeline.engine import SmartCartEngine
from smartcart.training.trainer import ModelTrainer


class RecommendationService:
    """Manages loaded recommendation models and handles API inference requests."""

    def __init__(self, data_dir: str = "data", checkpoint_dir: str = "checkpoints") -> None:
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.catalog = ItemCatalog(num_items=500, random_seed=42)
        self.preprocessor = InteractionPreprocessor()
        self.treatment_engine: Optional[SmartCartEngine] = None
        self.control_engine: Optional[SmartCartEngine] = None
        self.benchmark_data: List[Dict[str, Any]] = []

        self._initialize_service()

    def _initialize_service(self) -> None:
        cat_file = self.data_dir / "catalog.csv"
        if cat_file.exists():
            cat_df = pd.read_csv(cat_file)
            self.catalog = ItemCatalog(num_items=len(cat_df), random_seed=42)
            for row in cat_df.itertuples(index=False):
                complements = (
                    [int(c) for c in str(row.complements).split(",") if c]
                    if pd.notna(row.complements)
                    else []
                )
                prod = self.catalog.items.get(row.item_id)
                if prod:
                    prod.name = row.name
                    prod.category = row.category
                    prod.price = float(row.price)
                    prod.complements = complements

        vocab_file = self.checkpoint_dir / "vocab.json"
        ckpt_file = self.checkpoint_dir / "mf_model.pt"

        if vocab_file.exists() and ckpt_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            self.preprocessor.user_to_idx = {int(k): v for k, v in vocab["user_to_idx"].items()}
            self.preprocessor.idx_to_user = {v: int(k) for k, v in vocab["user_to_idx"].items()}
            self.preprocessor.item_to_idx = {int(k): v for k, v in vocab["item_to_idx"].items()}
            self.preprocessor.idx_to_item = {v: int(k) for k, v in vocab["item_to_idx"].items()}
            self.preprocessor.is_fitted = True

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            treatment_model = ModelTrainer.load_recommender(ckpt_file, device=device)

            train_file = self.data_dir / "train.csv"
            if train_file.exists():
                train_df = self.preprocessor.transform(pd.read_csv(train_file))
                control_model = PopularityRecommender(num_items=vocab["num_items"]).fit(train_df)
            else:
                control_model = PopularityRecommender(num_items=vocab["num_items"])

            self.control_engine = SmartCartEngine(
                model=control_model,
                catalog=self.catalog,
                preprocessor=self.preprocessor,
            )
            self.treatment_engine = SmartCartEngine(
                model=treatment_model,
                catalog=self.catalog,
                preprocessor=self.preprocessor,
                complement_weight=0.5,
            )
        else:
            # Cold-start fallback engine
            dummy_df = pd.DataFrame(
                {"user_id": [0, 1], "item_id": [0, 1], "timestamp": [0, 1]}
            )
            self.preprocessor.fit(dummy_df)
            fallback_model = PopularityRecommender(num_items=len(self.catalog.items))
            fallback_model.item_scores = np.ones(len(self.catalog.items), dtype=np.float32)
            fallback_model.is_fitted = True

            self.control_engine = SmartCartEngine(
                model=fallback_model,
                catalog=self.catalog,
                preprocessor=self.preprocessor,
            )
            self.treatment_engine = self.control_engine

        # Load offline benchmarks if available
        bench_file = Path("artifacts/benchmark_results.csv")
        if bench_file.exists():
            self.benchmark_data = pd.read_csv(bench_file).to_dict(orient="records")

    def get_catalog_items(
        self, category: Optional[str] = None, search: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        items = list(self.catalog.items.values())
        if category and category != "All":
            items = [i for i in items if i.category == category]
        if search:
            q = search.lower()
            items = [i for i in items if q in i.name.lower() or q in i.category.lower()]

        return [
            {
                "item_id": p.item_id,
                "name": p.name,
                "category": p.category,
                "price": p.price,
                "complements": p.complements,
            }
            for p in items[:limit]
        ]

    def get_recommendations(
        self, user_id: int, cart_item_ids: List[int], top_k: int = 4
    ) -> List[Dict[str, Any]]:
        engine = self.treatment_engine or self.control_engine
        if not engine:
            return []

        recs = engine.recommend_for_cart(
            user_id=user_id,
            cart_item_ids=cart_item_ids,
            top_k=top_k,
        )
        return [asdict(r) for r in recs]

    def run_live_ab_simulation(
        self, num_users: int = 5000, traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        if not self.control_engine or not self.treatment_engine:
            raise RuntimeError("Recommendation engines not initialized.")

        cfg = ABTestConfig(
            num_simulation_users=num_users,
            traffic_split=traffic_split,
            random_seed=int(np.random.randint(1, 10000)),
        )
        simulator = ABExperimentSimulator(
            control_engine=self.control_engine,
            treatment_engine=self.treatment_engine,
            catalog=self.catalog,
            config=cfg,
        )

        test_users = list(self.preprocessor.user_to_idx.keys()) or list(range(num_users))
        cohort = simulator.rng.choice(test_users, size=num_users, replace=True).tolist()

        records, summary = simulator.run_simulation(test_user_ids=cohort, top_k=4)

        # Compute sample distribution histogram for visualization
        ctrl_tx = [r.final_order_value for r in records if r.group == "control"]
        treat_tx = [r.final_order_value for r in records if r.group == "treatment"]

        return {
            "summary": asdict(summary),
            "sample_distributions": {
                "control_orders": ctrl_tx[:150],
                "treatment_orders": treat_tx[:150],
            },
        }
