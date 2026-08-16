"""Configuration schema and loader for the Smart Cart recommendation engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class DataConfig:
    num_users: int = 5000
    num_items: int = 500
    num_categories: int = 15
    num_interactions: int = 120000
    test_ratio: float = 0.15
    val_ratio: float = 0.10
    random_seed: int = 42
    data_dir: str = "data"


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_negatives: int = 4
    batch_size: int = 512
    num_epochs: int = 15
    dropout: float = 0.1
    top_k: int = 10


@dataclass
class ABTestConfig:
    num_simulation_users: int = 10000
    traffic_split: float = 0.5
    base_acceptance_rate: float = 0.08
    treatment_lift_factor: float = 1.35
    price_sensitivity_decay: float = 0.02
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        data_cfg = DataConfig(**raw_cfg.get("data", {}))
        model_cfg = ModelConfig(**raw_cfg.get("model", {}))
        ab_cfg = ABTestConfig(**raw_cfg.get("ab_test", {}))

        return cls(data=data_cfg, model=model_cfg, ab_test=ab_cfg)
