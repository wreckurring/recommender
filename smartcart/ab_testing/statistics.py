"""Statistical hypothesis testing and bootstrap confidence interval computation for A/B experiments."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import scipy.stats as stats


@dataclass
class HypothesisTestResult:
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift_pct: float
    p_value: float
    is_statistically_significant: bool
    ci_lower_pct: float
    ci_upper_pct: float


def compute_welch_ttest(
    control_samples: np.ndarray,
    treatment_samples: np.ndarray,
    alpha: float = 0.05,
    metric_name: str = "AOV",
) -> HypothesisTestResult:
    """Perform Welch's two-sample t-test (unequal variances) on continuous revenue distributions."""
    ctrl_mean = float(np.mean(control_samples))
    treat_mean = float(np.mean(treatment_samples))

    abs_lift = treat_mean - ctrl_mean
    rel_lift_pct = (abs_lift / ctrl_mean * 100.0) if ctrl_mean != 0 else 0.0

    t_stat, p_val = stats.ttest_ind(treatment_samples, control_samples, equal_var=False)

    # Bootstrap 95% Confidence Interval for Relative Lift (%)
    ci_lower, ci_upper = compute_bootstrap_relative_ci(
        control_samples=control_samples,
        treatment_samples=treatment_samples,
        num_bootstraps=2000,
        alpha=alpha,
    )

    return HypothesisTestResult(
        metric_name=metric_name,
        control_mean=ctrl_mean,
        treatment_mean=treat_mean,
        absolute_lift=abs_lift,
        relative_lift_pct=rel_lift_pct,
        p_value=float(p_val),
        is_statistically_significant=bool(p_val < alpha),
        ci_lower_pct=ci_lower,
        ci_upper_pct=ci_upper,
    )


def compute_bootstrap_relative_ci(
    control_samples: np.ndarray,
    treatment_samples: np.ndarray,
    num_bootstraps: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval for percentage lift: (treat_mean - ctrl_mean) / ctrl_mean * 100."""
    rng = np.random.default_rng(random_seed)
    n_ctrl = len(control_samples)
    n_treat = len(treatment_samples)

    boot_lifts = np.empty(num_bootstraps, dtype=np.float64)

    for i in range(num_bootstraps):
        c_boot = rng.choice(control_samples, size=n_ctrl, replace=True)
        t_boot = rng.choice(treatment_samples, size=n_treat, replace=True)
        c_m = np.mean(c_boot)
        t_m = np.mean(t_boot)
        boot_lifts[i] = ((t_m - c_m) / c_m) * 100.0 if c_m > 0 else 0.0

    lower_pct = float(np.percentile(boot_lifts, 100.0 * (alpha / 2.0)))
    upper_pct = float(np.percentile(boot_lifts, 100.0 * (1.0 - alpha / 2.0)))

    return lower_pct, upper_pct


def compute_proportions_ztest(
    count_ctrl: int,
    nobs_ctrl: int,
    count_treat: int,
    nobs_treat: int,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Two-proportion z-test for binary conversion metrics."""
    p_ctrl = count_ctrl / nobs_ctrl if nobs_ctrl > 0 else 0.0
    p_treat = count_treat / nobs_treat if nobs_treat > 0 else 0.0

    abs_lift = p_treat - p_ctrl
    rel_lift_pct = (abs_lift / p_ctrl * 100.0) if p_ctrl > 0 else 0.0

    p_pooled = (count_ctrl + count_treat) / (nobs_ctrl + nobs_treat)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / nobs_ctrl + 1 / nobs_treat))

    z_score = abs_lift / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Normal approximation CI
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt((p_ctrl * (1 - p_ctrl) / nobs_ctrl) + (p_treat * (1 - p_treat) / nobs_treat))
    ci_lower = ((abs_lift - z_crit * se_diff) / p_ctrl * 100.0) if p_ctrl > 0 else 0.0
    ci_upper = ((abs_lift + z_crit * se_diff) / p_ctrl * 100.0) if p_ctrl > 0 else 0.0

    return HypothesisTestResult(
        metric_name="Conversion Rate",
        control_mean=p_ctrl,
        treatment_mean=p_treat,
        absolute_lift=abs_lift,
        relative_lift_pct=rel_lift_pct,
        p_value=float(p_val),
        is_statistically_significant=bool(p_val < alpha),
        ci_lower_pct=ci_lower,
        ci_upper_pct=ci_upper,
    )
