#!/usr/bin/env python3
"""02_advanced_usage.py — Advanced features: streaming, statistics, effect sizes.

Demonstrates:
  - Streaming progress with run_stream()
  - Wilson CI and bootstrap CI comparison
  - Cohen's h effect sizes and z-test significance
  - Pairwise significance matrix across multiple models

Run:
    python3 examples/02_advanced_usage.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MOCK_MODE", "true")

from mobile_llm_benchmark import BenchmarkRunner, MODELS
from mobile_llm_benchmark.statistical import (
    wilson_ci, bootstrap_ci, cohens_h, effect_size_label,
    z_test_proportions, pairwise_significance,
)


def section(title):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ── 1. Streaming run ──────────────────────────────────────────────────────────
section("1. Streaming evaluation (run_stream)")

model_configs = [
    m for m in MODELS if m["name"] in ("Qwen2.5-3B", "SmolLM2-1.7B", "Llama-3.2-1B")
]
benchmark_ids = ["gsm8k", "arc_challenge"]

runner = BenchmarkRunner(mock_mode=True)
results = []
for partial, log_line in runner.run_stream(model_configs, benchmark_ids, n_samples=30):
    results = partial
    print(f"  {log_line}")

# ── 2. Wilson CI vs Bootstrap CI ──────────────────────────────────────────────
section("2. Wilson CI vs Bootstrap CI  (n=30, k=18 correct)")

import random; random.seed(42)
outcomes = [1] * 18 + [0] * 12
lo_w, acc_w, hi_w = wilson_ci(18, 30)
lo_b, acc_b, hi_b = bootstrap_ci(outcomes, n_bootstrap=2000, seed=42)
print(f"  Wilson    : acc={acc_w:.1%}  CI=[{lo_w:.4f}, {hi_w:.4f}]  width={hi_w-lo_w:.4f}")
print(f"  Bootstrap : acc={acc_b:.1%}  CI=[{lo_b:.4f}, {hi_b:.4f}]  width={hi_b-lo_b:.4f}")

# ── 3. Effect sizes ───────────────────────────────────────────────────────────
section("3. Cohen's h effect sizes")

comparisons = [
    (0.72, 0.55, "Phi-4-Mini vs Gemma-3-4B (GSM8K)"),
    (0.60, 0.36, "Qwen2.5-3B vs SmolLM2-1.7B (GSM8K)"),
    (0.55, 0.52, "Two very similar models"),
]
print(f"  {'Comparison':<45} {'h':>6}  Effect")
print("  " + "-" * 60)
for p1, p2, desc in comparisons:
    h = cohens_h(p1, p2)
    print(f"  {desc:<45} {h:>6.3f}  {effect_size_label(h)}")

# ── 4. Pairwise significance ──────────────────────────────────────────────────
section("4. Pairwise z-test significance (GSM8K)")

pairs = pairwise_significance(results, "gsm8k")
print(f"  {'Model A':<16}  {'Model B':<16}  {'Acc A':>6}  {'Acc B':>6}  {'h':>5}  {'p':>7}  Sig?")
print("  " + "-" * 72)
for row in pairs:
    print(f"  {row['model_a']:<16}  {row['model_b']:<16}  "
          f"{row['accuracy_a']:>6.1%}  {row['accuracy_b']:>6.1%}  "
          f"{row['cohens_h']:>5.3f}  {row['p_value']:>7.4f}  "
          f"{'YES' if row['significant'] else 'no'}")
