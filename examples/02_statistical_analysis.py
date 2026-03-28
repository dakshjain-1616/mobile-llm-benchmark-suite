"""
Example 02: Statistical Analysis
=================================
Shows how to use the built-in statistical toolkit to:
  - Compute Wilson 95% confidence intervals for any accuracy
  - Calculate Cohen's h effect size between two models
  - Run a two-proportion z-test for statistical significance
  - Perform a full pairwise significance analysis across benchmark results

Run:
    python3 examples/02_statistical_analysis.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MOCK_MODE", "true")

from mobile_llm_benchmark.statistical import (
    wilson_ci,
    bootstrap_ci,
    cohens_h,
    effect_size_label,
    z_test_proportions,
    pairwise_significance,
)
from mobile_llm_benchmark import BenchmarkRunner


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main():
    print("=" * 60)
    print("Mobile LLM Benchmark Suite — Statistical Analysis Example")
    print("=" * 60)

    # ── 1. Wilson Confidence Intervals ──────────────────────────────────────
    section("1. Wilson 95% Confidence Intervals")
    test_cases = [
        (30, 50,  "Qwen2.5-3B  on ARC  (50 samples)"),
        (14, 50,  "Llama-3.2-1B on GSM8K (50 samples)"),
        (18, 50,  "SmolLM2-1.7B on GSM8K (50 samples)"),
        (7,  20,  "Any model   (20 samples)"),
        (1,  10,  "Edge case   (very small n)"),
    ]
    print(f"  {'Description':<40} {'Acc':>6}  {'95% CI':<22}  {'Width':>7}")
    print("  " + "-" * 75)
    for n_correct, n, desc in test_cases:
        lo, acc, hi = wilson_ci(n_correct, n)
        width = hi - lo
        print(f"  {desc:<40} {acc:>6.1%}  [{lo:.3f}, {hi:.3f}]  {width:>7.3f}")

    # ── 2. Bootstrap CI vs Wilson CI ────────────────────────────────────────
    section("2. Bootstrap CI vs Wilson CI  (n=20, k=12 correct)")
    import random
    random.seed(42)
    outcomes = [1]*12 + [0]*8
    lo_w, acc_w, hi_w = wilson_ci(12, 20)
    lo_b, acc_b, hi_b = bootstrap_ci(outcomes, n_bootstrap=2000, seed=42)
    print(f"  Wilson    CI: [{lo_w:.4f}, {hi_w:.4f}]  width={hi_w-lo_w:.4f}")
    print(f"  Bootstrap CI: [{lo_b:.4f}, {hi_b:.4f}]  width={hi_b-lo_b:.4f}")
    print("  (Wilson preferred for small n / extreme proportions)")

    # ── 3. Effect Sizes ──────────────────────────────────────────────────────
    section("3. Cohen's h Effect Sizes")
    comparisons = [
        (0.72, 0.55, "Phi-4-Mini vs Gemma-3-4B  (GSM8K)"),
        (0.60, 0.36, "Qwen2.5-3B vs SmolLM2-1.7B (GSM8K)"),
        (0.72, 0.60, "HellaSwag top vs bottom"),
        (0.55, 0.52, "Two similar models"),
    ]
    print(f"  {'Comparison':<42} {'h':>6}  {'Effect'}")
    print("  " + "-" * 62)
    for p1, p2, desc in comparisons:
        h = cohens_h(p1, p2)
        label = effect_size_label(h)
        print(f"  {desc:<42} {h:>6.3f}  {label}")

    # ── 4. Z-test for Significance ──────────────────────────────────────────
    section("4. Two-Proportion Z-Test")
    pairs = [
        (30, 50, 18, 50, "Qwen2.5-3B vs SmolLM2-1.7B  (GSM8K, n=50)"),
        (21, 50, 20, 50, "Very similar models"),
        (36, 50, 15, 50, "Large gap example"),
    ]
    print(f"  {'Comparison':<50} {'z-stat':>7}  {'p-value':>8}  Significant?")
    print("  " + "-" * 80)
    for nca, na, ncb, nb, desc in pairs:
        z, p = z_test_proportions(nca, na, ncb, nb)
        sig = "YES ✓" if p < 0.05 else "no"
        print(f"  {desc:<50} {z:>7.3f}  {p:>8.4f}  {sig}")

    # ── 5. Full Pairwise Analysis from a Real Mock Run ───────────────────────
    section("5. Pairwise Significance (Full Benchmark Run)")
    runner = BenchmarkRunner(mock_mode=True)
    model_configs = [
        {"name": "Qwen2.5-3B",    "id": "Qwen/Qwen2.5-3B-Instruct",              "params": "3B",   "provider": "hf"},
        {"name": "SmolLM2-1.7B",  "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  "params": "1.7B", "provider": "hf"},
        {"name": "Llama-3.2-1B",  "id": "meta-llama/llama-3.2-1b-instruct",      "params": "1B",   "provider": "openrouter"},
    ]
    results = runner.run(model_configs=model_configs, benchmark_ids=["gsm8k", "arc_challenge"], n_samples=50)
    pairs_data = pairwise_significance(results, "gsm8k")

    print(f"  GSM8K pairwise comparisons:")
    print(f"  {'Model A':<16}  {'Model B':<16}  {'Acc A':>6}  {'Acc B':>6}  {'Δ':>5}  {'h':>5}  {'p':>7}  Sig?")
    print("  " + "-" * 82)
    for row in pairs_data:
        delta = row["accuracy_a"] - row["accuracy_b"]
        print(
            f"  {row['model_a']:<16}  {row['model_b']:<16}  "
            f"{row['accuracy_a']:>6.1%}  {row['accuracy_b']:>6.1%}  "
            f"{delta:>+5.1%}  {row['cohens_h']:>5.3f}  {row['p_value']:>7.4f}  "
            f"{'YES' if row['significant'] else 'no'}"
        )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
