#!/usr/bin/env python3
"""
Example 01: Quick Start
=======================
Demonstrates the fastest way to benchmark mobile LLMs using the built-in
mock mode: 2 benchmarks × 20 samples × 2 models, no API key needed.

Run:
    python3 examples/01_quick_start.py
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force mock mode so the example works offline / without API keys
os.environ.setdefault("MOCK_MODE", "true")

from mobile_llm_benchmark import BenchmarkRunner, MODELS


def main():
    print("=" * 60)
    print("Mobile LLM Benchmark Suite — Quick Start Example")
    print("=" * 60)
    print()
    print("Scenario : 2 benchmarks × 20 samples (mock mode, no API key)")
    print()

    # Filter MODELS list to only the two models we want
    model_configs = [m for m in MODELS if m["name"] in ("Llama-3.2-1B", "SmolLM2-1.7B")]
    benchmark_ids = ["arc_challenge", "hellaswag"]

    runner = BenchmarkRunner(mock_mode=True)
    results = runner.run(
        model_configs=model_configs,
        benchmark_ids=benchmark_ids,
        n_samples=20,
    )

    # ── Print results table ──────────────────────────────────────────────────
    print(f"{'Model':<18} {'Benchmark':<15} {'Accuracy':>9}  {'95% CI':<22}  Correct")
    print("-" * 75)
    for r in sorted(results, key=lambda r: (r.model_name, r.benchmark)):
        ci = f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]"
        print(f"{r.model_name:<18} {r.benchmark_name:<15} {r.accuracy:>9.1%}  {ci:<22}  "
              f"{r.n_correct}/{r.n_samples}")

    print()

    # ── Summary ─────────────────────────────────────────────────────────────
    model_avgs: dict[str, list[float]] = {}
    for r in results:
        model_avgs.setdefault(r.model_name, []).append(r.accuracy)

    print("Average accuracy per model:")
    for model, accs in sorted(model_avgs.items(), key=lambda kv: -sum(kv[1])):
        avg = sum(accs) / len(accs)
        bar = "█" * int(avg * 30)
        print(f"  {model:<20} {avg:.1%}  {bar}")

    print()
    print("Done! Run `python3 demo.py --scenario quick` for a richer CLI view.")


if __name__ == "__main__":
    main()
