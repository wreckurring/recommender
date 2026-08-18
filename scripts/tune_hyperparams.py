"""CLI script to run hyperparameter tuning sweeps over Matrix Factorization models."""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.training.tuner import HyperparameterTuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep to balance precision/recall.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing train.csv and val.csv")
    parser.add_argument("--num-trials", type=int, default=6, help="Number of random search trials")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff for evaluation metrics")
    parser.add_argument("--output-json", type=str, default="data/tuning_results.json", help="Path to output JSON")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    train_file = data_path / "train.csv"
    val_file = data_path / "val.csv"

    if not train_file.exists() or not val_file.exists():
        print(f"Data files not found in {data_path}. Run `python scripts/generate_data.py` first.")
        sys.exit(1)

    print("Loading datasets...")
    train_raw = pd.read_csv(train_file)
    val_raw = pd.read_csv(val_file)

    preprocessor = InteractionPreprocessor()
    train_df = preprocessor.fit_transform(train_raw)
    val_df = preprocessor.transform(val_raw)

    meta = preprocessor.get_metadata(train_df)
    user_positives = preprocessor.get_user_positive_items(train_df)

    print(f"Dataset stats: {meta.num_users} users, {meta.num_items} items, {meta.num_interactions} train interactions.")

    param_distributions = {
        "embedding_dim": [16, 32, 64, 128],
        "learning_rate": [0.005, 0.001, 0.0005],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "num_negatives": [1, 2, 4],
        "batch_size": [256, 512],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sweep on device: {device}...")

    tuner = HyperparameterTuner(
        train_df=train_df,
        val_df=val_df,
        num_users=meta.num_users,
        num_items=meta.num_items,
        user_positives=user_positives,
        device=device,
    )

    trials = tuner.run_random_search(
        param_distributions=param_distributions,
        num_trials=args.num_trials,
        k=args.k,
        epochs_per_trial=args.epochs,
    )

    best_trial = tuner.get_best_trial(trials, metric="f1_at_k")

    print("\n--- Hyperparameter Sweep Summary ---")
    print(f"Best Trial #{best_trial.trial_id} (Optimized for F1@{args.k}):")
    print(f"  Params: {best_trial.params}")
    print(f"  Precision@{args.k}: {best_trial.precision_at_k:.4f}")
    print(f"  Recall@{args.k}:    {best_trial.recall_at_k:.4f}")
    print(f"  F1@{args.k}:        {best_trial.f1_at_k:.4f}")
    print(f"  NDCG@{args.k}:      {best_trial.ndcg_at_k:.4f}")

    # Export results
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial": {
                    "trial_id": best_trial.trial_id,
                    "params": best_trial.params,
                    "precision_at_k": best_trial.precision_at_k,
                    "recall_at_k": best_trial.recall_at_k,
                    "f1_at_k": best_trial.f1_at_k,
                    "ndcg_at_k": best_trial.ndcg_at_k,
                },
                "all_trials": [
                    {
                        "trial_id": t.trial_id,
                        "params": t.params,
                        "precision_at_k": t.precision_at_k,
                        "recall_at_k": t.recall_at_k,
                        "f1_at_k": t.f1_at_k,
                        "ndcg_at_k": t.ndcg_at_k,
                    }
                    for t in trials
                ],
            },
            f,
            indent=2,
        )
    print(f"\nTuning results exported to {out_file.resolve()}")


if __name__ == "__main__":
    main()
