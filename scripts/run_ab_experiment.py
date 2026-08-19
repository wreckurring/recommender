"""CLI script to run simulated A/B testing checkout experiments and analyze business impact."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import pandas as pd
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from smartcart.ab_testing.simulator import ABExperimentSimulator
from smartcart.config import ExperimentConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.models.baselines import PopularityRecommender
from smartcart.models.matrix_factorization import MatrixFactorizationRecommender
from smartcart.pipeline.engine import SmartCartEngine
from smartcart.training.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B experiment simulation and evaluate business KPIs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing catalog.csv and test.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mf_model.pt", help="Path to trained model")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json", help="Path to preprocessor vocab")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save experiment results")
    parser.add_argument("--users", type=int, default=10000, help="Number of simulated users in experiment")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = ExperimentConfig.from_yaml(cfg_path) if cfg_path.exists() else ExperimentConfig()
    cfg.ab_test.num_simulation_users = args.users

    data_dir = Path(args.data_dir)
    cat_file = data_dir / "catalog.csv"
    test_file = data_dir / "test.csv"
    train_file = data_dir / "train.csv"
    ckpt_path = Path(args.checkpoint)
    vocab_path = Path(args.vocab)

    if not cat_file.exists() or not ckpt_path.exists() or not vocab_path.exists():
        print("Required artifacts not found. Please run `scripts/generate_data.py` and `scripts/train.py` first.")
        sys.exit(1)

    print("--- 1. Loading Catalog and Model Checkpoint ---")
    cat_df = pd.read_csv(cat_file)
    catalog = ItemCatalog(num_items=len(cat_df), random_seed=cfg.data.random_seed)

    # Reconstruct catalog from CSV
    for row in cat_df.itertuples(index=False):
        complements = [int(c) for c in str(row.complements).split(",") if c] if pd.notna(row.complements) else []
        prod = catalog.items.get(row.item_id)
        if prod:
            prod.name = row.name
            prod.category = row.category
            prod.price = float(row.price)
            prod.complements = complements

    # Load vocab and preprocessor
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    preprocessor = InteractionPreprocessor()
    preprocessor.user_to_idx = {int(k): v for k, v in vocab["user_to_idx"].items()}
    preprocessor.idx_to_user = {v: int(k) for k, v in vocab["user_to_idx"].items()}
    preprocessor.item_to_idx = {int(k): v for k, v in vocab["item_to_idx"].items()}
    preprocessor.idx_to_item = {v: int(k) for k, v in vocab["item_to_idx"].items()}
    preprocessor.is_fitted = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    treatment_model = ModelTrainer.load_recommender(ckpt_path, device=device)

    # Control baseline: Popularity Recommender
    train_df = preprocessor.transform(pd.read_csv(train_file))
    control_model = PopularityRecommender(num_items=vocab["num_items"]).fit(train_df)

    control_engine = SmartCartEngine(
        model=control_model,
        catalog=catalog,
        preprocessor=preprocessor,
    )
    treatment_engine = SmartCartEngine(
        model=treatment_model,
        catalog=catalog,
        preprocessor=preprocessor,
        complement_weight=0.5,
    )

    print("--- 2. Executing Randomized A/B Checkout Simulation ---")
    print(f"Simulating {cfg.ab_test.num_simulation_users:,} user transactions (50% Control, 50% Treatment)...")

    simulator = ABExperimentSimulator(
        control_engine=control_engine,
        treatment_engine=treatment_engine,
        catalog=catalog,
        config=cfg.ab_test,
    )

    # Generate user cohort
    test_users = list(preprocessor.user_to_idx.keys())
    rng = simulator.rng
    cohort = rng.choice(test_users, size=cfg.ab_test.num_simulation_users, replace=True).tolist()

    records, summary = simulator.run_simulation(test_user_ids=cohort, top_k=4)

    print("\n============================================================")
    print("               A/B TEST EXPERIMENT SUMMARY")
    print("============================================================")
    print(f"Total Transactions:        {len(records):,}")
    print(f"Control Traffic:           {summary.num_users_control:,} users")
    print(f"Treatment Traffic:         {summary.num_users_treatment:,} users")
    print("------------------------------------------------------------")
    print(f"Control AOV:               ${summary.control_aov:.2f}")
    print(f"Treatment AOV:             ${summary.treatment_aov:.2f}")
    print(f"AOV Relative Lift:         +{summary.aov_relative_lift_pct:.2f}%")
    print(f"AOV 95% Confidence Interval: [{summary.aov_ci_95[0]:.2f}%, {summary.aov_ci_95[1]:.2f}%]")
    print(f"AOV p-value (Welch's t):   {summary.aov_p_value:.4e} (Significant: {summary.aov_statistically_significant})")
    print("------------------------------------------------------------")
    print(f"Control CVR (Cross-Sell):  {summary.control_cvr * 100:.2f}%")
    print(f"Treatment CVR:             {summary.treatment_cvr * 100:.2f}%")
    print(f"CVR Relative Lift:         +{summary.cvr_relative_lift_pct:.2f}% (p = {summary.cvr_p_value:.4e})")
    print("------------------------------------------------------------")
    print(f"Control Basket Size (UPT): {summary.control_upt:.2f} items")
    print(f"Treatment Basket Size:     {summary.treatment_upt:.2f} items (+{summary.upt_relative_lift_pct:.2f}%)")
    print("------------------------------------------------------------")
    print(f"Control Total Revenue:     ${summary.control_total_revenue:,.2f}")
    print(f"Treatment Total Revenue:   ${summary.treatment_total_revenue:,.2f}")
    print(f"Total Revenue Lift:        +{summary.total_revenue_lift_pct:.2f}%")
    print("============================================================\n")

    # Export artifacts
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    summary_dict = asdict(summary)
    with open(art_dir / "ab_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    df_tx = pd.DataFrame([asdict(r) for r in records])
    df_tx.to_csv(art_dir / "transactions.csv", index=False)

    print(f"Experiment summary saved to: {art_dir / 'ab_experiment_summary.json'}")
    print(f"Transaction logs saved to:   {art_dir / 'transactions.csv'}\n")


if __name__ == "__main__":
    main()
