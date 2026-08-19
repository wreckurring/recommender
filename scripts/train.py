"""End-to-end training, validation, and benchmarking pipeline CLI."""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from smartcart.config import ExperimentConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.data.dataset import BPRDataset
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.evaluation.evaluator import ModelEvaluator
from smartcart.models.baselines import ItemCooccurrenceRecommender, PopularityRecommender
from smartcart.models.matrix_factorization import MatrixFactorization, MatrixFactorizationRecommender
from smartcart.training.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Matrix Factorization model and benchmark against baselines.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset CSVs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save benchmark metrics")
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    cfg = ExperimentConfig.from_yaml(cfg_path) if cfg_path.exists() else ExperimentConfig()

    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"
    catalog_file = data_dir / "catalog.csv"

    if not train_file.exists() or not test_file.exists():
        print(f"Data files not found in {data_dir}. Run `python scripts/generate_data.py` first.")
        sys.exit(1)

    print("--- 1. Loading Datasets and Product Catalog ---")
    train_raw = pd.read_csv(train_file)
    val_raw = pd.read_csv(val_file) if val_file.exists() else None
    test_raw = pd.read_csv(test_file)

    if catalog_file.exists():
        cat_df = pd.read_csv(catalog_file)
        item_to_cat = dict(zip(cat_df["item_id"], cat_df["category"]))
    else:
        item_to_cat = {}

    preprocessor = InteractionPreprocessor()
    train_df = preprocessor.fit_transform(train_raw)
    val_df = preprocessor.transform(val_raw) if val_raw is not None else None
    test_df = preprocessor.transform(test_raw)

    meta = preprocessor.get_metadata(train_df)
    user_positives = preprocessor.get_user_positive_items(train_df)
    print(f"Entities: {meta.num_users:,} users | {meta.num_items:,} items | {meta.num_interactions:,} train interactions")
    print(f"Interaction Matrix Density: {meta.density * 100:.3f}%\n")

    print("--- 2. Initializing Datasets and PyTorch DataLoader ---")
    train_dataset = BPRDataset(
        user_indices=train_df["user_idx"].values,
        item_indices=train_df["item_idx"].values,
        num_items=meta.num_items,
        user_positives=user_positives,
        num_negatives=cfg.model.num_negatives,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)

    val_loader = None
    if val_df is not None:
        val_dataset = BPRDataset(
            user_indices=val_df["user_idx"].values,
            item_indices=val_df["item_idx"].values,
            num_items=meta.num_items,
            user_positives=user_positives,
            num_negatives=1,
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.model.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}\n")

    print("--- 3. Training Matrix Factorization (BPR-MF) ---")
    mf_model = MatrixFactorization(
        num_users=meta.num_users,
        num_items=meta.num_items,
        embedding_dim=cfg.model.embedding_dim,
    ).to(device)

    trainer = ModelTrainer(model=mf_model, config=cfg.model, device=device)
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, verbose=True)

    # Save checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = ckpt_dir / "mf_model.pt"
    trainer.save_checkpoint(model_save_path)

    # Save preprocessor index mapping
    with open(ckpt_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "user_to_idx": {str(k): v for k, v in preprocessor.user_to_idx.items()},
                "item_to_idx": {str(k): v for k, v in preprocessor.item_to_idx.items()},
                "num_users": meta.num_users,
                "num_items": meta.num_items,
            },
            f,
        )
    print(f"\nModel checkpoint and vocabulary saved to {ckpt_dir.resolve()}\n")

    print("--- 4. Benchmarking Models on Test Set ---")
    mf_recommender = MatrixFactorizationRecommender(
        num_users=meta.num_users,
        num_items=meta.num_items,
        embedding_dim=cfg.model.embedding_dim,
        device=str(device),
    )
    mf_recommender.model = mf_model
    mf_recommender.is_fitted = True

    pop_recommender = PopularityRecommender(num_items=meta.num_items)
    pop_recommender.fit(train_df)

    cooc_recommender = ItemCooccurrenceRecommender(num_items=meta.num_items)
    cooc_recommender.fit(train_df)

    evaluator = ModelEvaluator(
        test_df=test_df,
        num_items=meta.num_items,
        item_to_category=item_to_cat,
        train_user_positives=user_positives,
    )

    models = [pop_recommender, cooc_recommender, mf_recommender]
    benchmark_df = evaluator.benchmark_models(models=models, k_values=[5, 10, 20])

    print("\n--- Offline Ranking Benchmark Results ---")
    print(benchmark_df.to_string(index=False))

    # Save benchmark artifact
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(art_dir / "benchmark_results.csv", index=False)
    print(f"\nBenchmark metrics saved to {art_dir / 'benchmark_results.csv'}\n")


if __name__ == "__main__":
    main()
