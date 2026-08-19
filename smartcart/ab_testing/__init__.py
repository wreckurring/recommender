"""Strategic A/B testing and business impact evaluation framework."""

from smartcart.ab_testing.simulator import (
    ABExperimentSimulator,
    ExperimentSummary,
    TransactionRecord,
)
from smartcart.ab_testing.statistics import (
    HypothesisTestResult,
    compute_bootstrap_relative_ci,
    compute_proportions_ztest,
    compute_welch_ttest,
)

__all__ = [
    "ABExperimentSimulator",
    "ExperimentSummary",
    "TransactionRecord",
    "HypothesisTestResult",
    "compute_welch_ttest",
    "compute_bootstrap_relative_ci",
    "compute_proportions_ztest",
]
