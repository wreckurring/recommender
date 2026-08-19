"""Unit tests for A/B testing statistical hypothesis tests and checkout simulation."""

import unittest
import numpy as np
import pandas as pd

from smartcart.ab_testing.simulator import ABExperimentSimulator, ExperimentSummary
from smartcart.ab_testing.statistics import (
    compute_bootstrap_relative_ci,
    compute_proportions_ztest,
    compute_welch_ttest,
)
from smartcart.config import ABTestConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.data.preprocessor import InteractionPreprocessor
from smartcart.models.baselines import PopularityRecommender
from smartcart.pipeline.engine import SmartCartEngine


class TestABTesting(unittest.TestCase):
    def test_welch_ttest_and_bootstrap_ci(self):
        rng = np.random.default_rng(42)
        ctrl = rng.normal(loc=100.0, scale=15.0, size=500)
        treat = rng.normal(loc=110.0, scale=18.0, size=500)  # ~10% lift

        res = compute_welch_ttest(ctrl, treat, alpha=0.05, metric_name="AOV")
        self.assertGreater(res.relative_lift_pct, 5.0)
        self.assertLess(res.p_value, 0.01)
        self.assertTrue(res.is_statistically_significant)
        self.assertLess(res.ci_lower_pct, res.ci_upper_pct)

    def test_proportions_ztest(self):
        res = compute_proportions_ztest(
            count_ctrl=50, nobs_ctrl=1000, count_treat=80, nobs_treat=1000
        )
        self.assertAlmostEqual(res.control_mean, 0.05)
        self.assertAlmostEqual(res.treatment_mean, 0.08)
        self.assertGreater(res.relative_lift_pct, 0.0)
        self.assertTrue(res.is_statistically_significant)

    def test_ab_simulator_run(self):
        catalog = ItemCatalog(num_items=30, random_seed=42)
        df = pd.DataFrame(
            {
                "user_id": list(range(10)),
                "item_id": list(range(10)),
                "timestamp": list(range(10)),
            }
        )
        prep = InteractionPreprocessor().fit(df)
        model = PopularityRecommender(num_items=30).fit(prep.transform(df))

        engine = SmartCartEngine(model=model, catalog=catalog, preprocessor=prep)
        cfg = ABTestConfig(num_simulation_users=50, random_seed=42)

        simulator = ABExperimentSimulator(
            control_engine=engine,
            treatment_engine=engine,
            catalog=catalog,
            config=cfg,
        )

        records, summary = simulator.run_simulation(test_user_ids=list(range(50)))
        self.assertEqual(len(records), 50)
        self.assertIsInstance(summary, ExperimentSummary)
        self.assertGreater(summary.num_users_control, 0)
        self.assertGreater(summary.num_users_treatment, 0)


if __name__ == "__main__":
    unittest.main()
