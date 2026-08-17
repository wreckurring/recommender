"""Script to generate synthetic e-commerce catalog and 100K+ checkout interaction dataset."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from smartcart.config import ExperimentConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.data.generator import InteractionGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic checkout interactions and product catalog.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for CSV artifacts")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        cfg = ExperimentConfig.from_yaml(config_path)
    else:
        cfg = ExperimentConfig()

    print(f"Initializing catalog with {cfg.data.num_items} items across {cfg.data.num_categories} categories...")
    catalog = ItemCatalog(num_items=cfg.data.num_items, random_seed=cfg.data.random_seed)

    print(f"Simulating {cfg.data.num_interactions} checkout interactions for {cfg.data.num_users} users...")
    generator = InteractionGenerator(catalog=catalog, config=cfg.data)
    train_df, val_df, test_df = generator.generate_and_save(args.output_dir)

    print("\nData generation complete!")
    print(f"  Catalog items: {len(catalog.items)}")
    print(f"  Train interactions: {len(train_df)}")
    print(f"  Validation interactions: {len(val_df)}")
    print(f"  Test interactions: {len(test_df)}")
    print(f"  Artifacts saved to: {Path(args.output_dir).resolve()}\n")


if __name__ == "__main__":
    main()
