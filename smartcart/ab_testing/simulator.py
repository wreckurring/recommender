"""Checkout A/B experiment simulator modeling user purchase decisions and revenue lift."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from smartcart.ab_testing.statistics import (
    HypothesisTestResult,
    compute_proportions_ztest,
    compute_welch_ttest,
)
from smartcart.config import ABTestConfig
from smartcart.data.catalog import ItemCatalog
from smartcart.pipeline.engine import SmartCartEngine


@dataclass
class TransactionRecord:
    transaction_id: int
    user_id: int
    group: str  # "control" or "treatment"
    initial_cart_size: int
    final_cart_size: int
    initial_order_value: float
    final_order_value: float
    accepted_recommendation: bool
    accepted_item_id: Optional[int]
    accepted_item_price: float


@dataclass
class ExperimentSummary:
    num_users_control: int
    num_users_treatment: int
    control_aov: float
    treatment_aov: float
    aov_relative_lift_pct: float
    aov_p_value: float
    aov_statistically_significant: bool
    aov_ci_95: Tuple[float, float]
    control_cvr: float
    treatment_cvr: float
    cvr_relative_lift_pct: float
    cvr_p_value: float
    control_upt: float
    treatment_upt: float
    upt_relative_lift_pct: float
    control_total_revenue: float
    treatment_total_revenue: float
    total_revenue_lift_pct: float


class ABExperimentSimulator:
    """Simulates online checkout traffic and evaluates business impact on AOV, UPT, and CVR."""

    def __init__(
        self,
        control_engine: SmartCartEngine,
        treatment_engine: SmartCartEngine,
        catalog: ItemCatalog,
        config: ABTestConfig,
    ) -> None:
        self.control_engine = control_engine
        self.treatment_engine = treatment_engine
        self.catalog = catalog
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def _simulate_user_decision(
        self,
        recs: list,
        cart_value: float,
        is_treatment: bool,
    ) -> Tuple[bool, Optional[int], float]:
        """Model probabilistic user acceptance based on recommendation relevance and price."""
        if not recs:
            return False, None, 0.0

        for rec in recs:
            # Base acceptance propensity
            base_prob = self.config.base_acceptance_rate
            if is_treatment:
                # Treatment smart-cart model offers context-relevant complement boost
                base_prob *= self.config.treatment_lift_factor

            # Price resistance: higher price relative to current cart dampens acceptance
            price_ratio = rec.price / max(10.0, cart_value)
            price_penalty = np.exp(-self.config.price_sensitivity_decay * price_ratio * 100.0)
            
            # Acceptance probability bounded between 0.01 and 0.45
            prob = float(np.clip(base_prob * price_penalty, 0.01, 0.45))

            if self.rng.random() < prob:
                return True, rec.item_id, rec.price

        return False, None, 0.0

    def run_simulation(
        self,
        test_user_ids: List[int],
        initial_baskets: Optional[List[List[int]]] = None,
        top_k: int = 4,
    ) -> Tuple[List[TransactionRecord], ExperimentSummary]:
        """Execute randomized checkout sessions across Control and Treatment groups."""
        all_item_ids = list(self.catalog.items.keys())
        num_users = min(len(test_user_ids), self.config.num_simulation_users)
        records: List[TransactionRecord] = []

        for tx_id in range(num_users):
            user_id = test_user_ids[tx_id]
            
            # Generate initial organic basket
            if initial_baskets and tx_id < len(initial_baskets):
                cart = list(initial_baskets[tx_id])
            else:
                basket_len = int(self.rng.integers(2, 5))
                cart = self.rng.choice(all_item_ids, size=basket_len, replace=False).tolist()

            initial_cart_size = len(cart)
            initial_val = sum(self.catalog.get_price(i) for i in cart)

            # Random 50/50 A/B Traffic Assignment
            is_treatment = bool(self.rng.random() < self.config.traffic_split)
            group_label = "treatment" if is_treatment else "control"
            engine = self.treatment_engine if is_treatment else self.control_engine

            recs = engine.recommend_for_cart(
                user_id=user_id,
                cart_item_ids=cart,
                top_k=top_k,
            )

            accepted, item_id, item_price = self._simulate_user_decision(
                recs=recs,
                cart_value=initial_val,
                is_treatment=is_treatment,
            )

            final_cart_size = initial_cart_size + (1 if accepted else 0)
            final_val = initial_val + (item_price if accepted else 0.0)

            records.append(
                TransactionRecord(
                    transaction_id=tx_id,
                    user_id=user_id,
                    group=group_label,
                    initial_cart_size=initial_cart_size,
                    final_cart_size=final_cart_size,
                    initial_order_value=float(np.round(initial_val, 2)),
                    final_order_value=float(np.round(final_val, 2)),
                    accepted_recommendation=accepted,
                    accepted_item_id=item_id,
                    accepted_item_price=float(np.round(item_price, 2)),
                )
            )

        summary = self._compute_summary_statistics(records)
        return records, summary

    def _compute_summary_statistics(self, records: List[TransactionRecord]) -> ExperimentSummary:
        df = pd.DataFrame([asdict(r) for r in records])
        ctrl = df[df["group"] == "control"]
        treat = df[df["group"] == "treatment"]

        # AOV stats
        ctrl_revs = ctrl["final_order_value"].values
        treat_revs = treat["final_order_value"].values
        aov_test = compute_welch_ttest(ctrl_revs, treat_revs, metric_name="AOV")

        # CVR stats
        cvr_test = compute_proportions_ztest(
            count_ctrl=int(ctrl["accepted_recommendation"].sum()),
            nobs_ctrl=len(ctrl),
            count_treat=int(treat["accepted_recommendation"].sum()),
            nobs_treat=len(treat),
        )

        # UPT (Units per Transaction)
        ctrl_upt = float(ctrl["final_cart_size"].mean())
        treat_upt = float(treat["final_cart_size"].mean())
        upt_lift = ((treat_upt - ctrl_upt) / ctrl_upt * 100.0) if ctrl_upt > 0 else 0.0

        # Total revenue
        ctrl_tot = float(ctrl_revs.sum())
        treat_tot = float(treat_revs.sum())
        tot_rev_lift = ((treat_tot - ctrl_tot) / ctrl_tot * 100.0) if ctrl_tot > 0 else 0.0

        return ExperimentSummary(
            num_users_control=len(ctrl),
            num_users_treatment=len(treat),
            control_aov=aov_test.control_mean,
            treatment_aov=aov_test.treatment_mean,
            aov_relative_lift_pct=aov_test.relative_lift_pct,
            aov_p_value=aov_test.p_value,
            aov_statistically_significant=aov_test.is_statistically_significant,
            aov_ci_95=(aov_test.ci_lower_pct, aov_test.ci_upper_pct),
            control_cvr=cvr_test.control_mean,
            treatment_cvr=cvr_test.treatment_mean,
            cvr_relative_lift_pct=cvr_test.relative_lift_pct,
            cvr_p_value=cvr_test.p_value,
            control_upt=ctrl_upt,
            treatment_upt=treat_upt,
            upt_relative_lift_pct=upt_lift,
            control_total_revenue=ctrl_tot,
            treatment_total_revenue=treat_tot,
            total_revenue_lift_pct=tot_rev_lift,
        )
