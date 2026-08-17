"""Data generation, catalog structures, and dataset abstractions for Smart Cart."""

from smartcart.data.catalog import ItemCatalog, Product
from smartcart.data.dataset import BPRDataset, NegativeSampler, PointwiseDataset
from smartcart.data.generator import InteractionGenerator
from smartcart.data.preprocessor import InteractionPreprocessor, DatasetMetadata

__all__ = [
    "ItemCatalog",
    "Product",
    "InteractionGenerator",
    "InteractionPreprocessor",
    "DatasetMetadata",
    "BPRDataset",
    "PointwiseDataset",
    "NegativeSampler",
]
