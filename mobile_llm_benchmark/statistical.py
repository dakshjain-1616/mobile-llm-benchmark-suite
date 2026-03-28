"""Statistical utilities: Wilson CI, bootstrap CI, Cohen's h, z-test, effect sizes."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Returns (ci_lower, accuracy, ci_upper) where accuracy = successes / n.
    All values are in [0, 1].

    The Wilson interval is preferred over the Wald interval because it has
    better coverage for small n and extreme proportions.
    """
    if n <= 0:
        return 0.0, 0.0, 0.0

    p = successes / n
    z = float(stats.norm.ppf((1.0 + confidence) / 2.0))
    z2 = z * z

    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom

    ci_lower = 0.0 if successes == 0 else float(max(0.0, center - margin))
    ci_upper = 1.0 if successes == n else float(min(1.0, center + margin))
    return ci_lower, float(p), ci_upper


def bootstrap_ci(
    outcomes: list[int],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for accuracy.

    Args:
        outcomes: List of 0/1 values (0=wrong, 1=correct).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples (default 2000).
        seed: RNG seed for reproducibility.

    Returns:
        (ci_lower, accuracy, ci_upper)
    """
    if not outcomes:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    arr = np.array(outcomes, dtype=float)
    accuracy = float(arr.mean())

    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return float(np.clip(ci_lower, 0.0, 1.0)), accuracy, float(np.clip(ci_upper, 0.0, 1.0))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for comparing two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    |h| < 0.2  → negligible
    |h| < 0.5  → small
    |h| < 0.8  → medium
    |h| >= 0.8 → large
    """
    phi1 = 2.0 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0)))
    phi2 = 2.0 * np.arcsin(np.sqrt(np.clip(p2, 0.0, 1.0)))
    return float(abs(phi1 - phi2))


def effect_size_label(h: float) -> str:
    """Convert Cohen's h to a human-readable label."""
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    return "large"


def z_test_proportions(
    n_correct_a: int,
    n_a: int,
    n_correct_b: int,
    n_b: int,
) -> tuple[float, float]:
    """Two-proportion z-test.

    Tests H0: p_A == p_B against H1: p_A != p_B.

    Returns:
        (z_statistic, p_value) — two-tailed.
        p_value < 0.05 → statistically significant difference at 95% confidence.
    """
    if n_a <= 0 or n_b <= 0:
        return 0.0, 1.0

    p_a = n_correct_a / n_a
    p_b = n_correct_b / n_b
    # Pooled proportion under H0
    p_pool = (n_correct_a + n_correct_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_a - p_b) / se
    p_value = float(2 * stats.norm.sf(abs(z)))
    return float(z), p_value


def pairwise_significance(
    results: list,  # list of BenchmarkResult
    benchmark_id: str,
) -> list[dict]:
    """Compute pairwise significance tests for all model pairs on one benchmark.

    Returns a list of dicts with keys:
        model_a, model_b, accuracy_a, accuracy_b,
        cohens_h, effect_size, z_stat, p_value, significant
    """
    relevant = [r for r in results if r.benchmark == benchmark_id]

    pairs = []
    for i, r1 in enumerate(relevant):
        for r2 in relevant[i + 1:]:
            z, p = z_test_proportions(r1.n_correct, r1.n_samples, r2.n_correct, r2.n_samples)
            h = cohens_h(r1.accuracy, r2.accuracy)
            pairs.append({
                "model_a": r1.model_name,
                "model_b": r2.model_name,
                "accuracy_a": round(r1.accuracy, 4),
                "accuracy_b": round(r2.accuracy, 4),
                "cohens_h": round(h, 4),
                "effect_size": effect_size_label(h),
                "z_stat": round(z, 4),
                "p_value": round(p, 4),
                "significant": p < 0.05,
            })
    return pairs


def pairwise_effects(
    results: list[dict],
    metric_key: str = "accuracy",
) -> list[dict]:
    """Compute pairwise Cohen's h between all model pairs for a given benchmark."""
    pairs = []
    for i, r1 in enumerate(results):
        for r2 in results[i + 1:]:
            h = cohens_h(r1[metric_key], r2[metric_key])
            pairs.append(
                {
                    "model_a": r1["model_name"],
                    "model_b": r2["model_name"],
                    "cohens_h": round(h, 4),
                    "effect_size": effect_size_label(h),
                }
            )
    return pairs


def aggregate_scores(
    results: list,  # list of BenchmarkResult
    model_names: list[str],
    benchmark_ids: list[str],
) -> dict[str, float]:
    """Return average accuracy per model across all benchmarks."""
    from collections import defaultdict

    sums: dict[str, list[float]] = defaultdict(list)
    for r in results:
        sums[r.model_name].append(r.accuracy)
    return {name: float(np.mean(vals)) if vals else 0.0 for name, vals in sums.items()}
