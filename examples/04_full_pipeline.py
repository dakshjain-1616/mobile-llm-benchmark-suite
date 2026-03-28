#!/usr/bin/env python3
"""04_full_pipeline.py — End-to-end workflow: run → analyse → report.

Demonstrates the complete pipeline:
  1. Configure models and benchmarks from SCENARIO_PRESETS
  2. Run evaluation with streaming progress
  3. Print a leaderboard with Wilson CIs
  4. Compute pairwise statistical significance
  5. Generate HTML + CSV + JSON reports to a temp directory
  6. Show an excerpt from the JSON report

Run:
    python3 examples/04_full_pipeline.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import json
import tempfile

from mobile_llm_benchmark import BenchmarkRunner, MODELS, BENCHMARKS
from mobile_llm_benchmark.config import SCENARIO_PRESETS, MODEL_BY_NAME
from mobile_llm_benchmark.statistical import pairwise_significance, aggregate_scores
from mobile_llm_benchmark.report_generator import ReportGenerator


def divider(title=""):
    width = 65
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print("═" * width)


# ── Step 1: choose 'math' preset ─────────────────────────────────────────────
divider("Step 1: Configuration")

preset = SCENARIO_PRESETS["math"]
print(f"  Preset    : {preset['name']}  —  {preset['description']}")

model_configs = [MODEL_BY_NAME[n] for n in preset["models"] if n in MODEL_BY_NAME]
benchmark_ids = preset["benchmarks"]
n_samples = 30  # reduced from preset default for fast demo

print(f"  Models    : {', '.join(m['name'] for m in model_configs)}")
print(f"  Benchmarks: {', '.join(benchmark_ids)}")
print(f"  Samples   : {n_samples} per benchmark")

# ── Step 2: streaming evaluation ─────────────────────────────────────────────
divider("Step 2: Evaluation")

runner = BenchmarkRunner(mock_mode=True)
results = []
total = len(model_configs) * len(benchmark_ids)

for partial, log_line in runner.run_stream(model_configs, benchmark_ids, n_samples=n_samples):
    results = partial
    done = len(results)
    pct = done / total if total else 0
    bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
    print(f"\r  [{bar}] {done:2d}/{total}  {log_line[:42]:<42}", end="", flush=True)

print(f"\r  [{'█' * 30}] {total}/{total}  Done!                            ")

# ── Step 3: leaderboard ───────────────────────────────────────────────────────
divider("Step 3: Leaderboard")

avg_scores = aggregate_scores(results, [m["name"] for m in model_configs], benchmark_ids)
ranked = sorted(avg_scores.items(), key=lambda kv: -kv[1])

print(f"  {'Rank':<5} {'Model':<18} {'Avg Accuracy':>13}  Chart")
print("  " + "-" * 60)
for rank, (model_name, avg_acc) in enumerate(ranked, 1):
    bar = "█" * int(avg_acc * 30)
    print(f"  #{rank:<4} {model_name:<18} {avg_acc:>13.1%}  {bar}")

# ── Step 4: pairwise significance (first benchmark) ──────────────────────────
divider("Step 4: Pairwise Significance")

bench_id = benchmark_ids[0]
pairs = pairwise_significance(results, bench_id)
bench_name = next((r.benchmark_name for r in results if r.benchmark == bench_id), bench_id)
print(f"  {bench_name} — pairwise z-tests:")
print(f"  {'Model A':<16}  {'Model B':<16}  {'ΔAcc':>6}  {'h':>5}  {'p-value':>8}  Sig?")
print("  " + "-" * 64)
for row in pairs:
    delta = row["accuracy_a"] - row["accuracy_b"]
    print(f"  {row['model_a']:<16}  {row['model_b']:<16}  "
          f"{delta:>+6.1%}  {row['cohens_h']:>5.3f}  {row['p_value']:>8.4f}  "
          f"{'YES ✓' if row['significant'] else 'no'}")

# ── Step 5: report generation ─────────────────────────────────────────────────
divider("Step 5: Report Generation")

with tempfile.TemporaryDirectory() as tmpdir:
    gen = ReportGenerator(output_dir=tmpdir)
    artifacts = gen.generate_all(results)

    print("  Generated artifacts:")
    for name, path in artifacts.items():
        if path and os.path.exists(path):
            kb = os.path.getsize(path) / 1024
            print(f"    ✓  {name:<22} {kb:>8.1f} KB  ({os.path.basename(path)})")

    # ── Step 6: JSON excerpt ──────────────────────────────────────────────────
    divider("Step 6: JSON Report Excerpt")

    json_path = artifacts.get("json")
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            report = json.load(f)

        print(f"  generated_at : {report.get('generated_at', 'n/a')}")
        print(f"  mock_mode    : {report.get('mock_mode', 'n/a')}")
        print(f"  n_models     : {report.get('n_models', 'n/a')}")
        print(f"  n_benchmarks : {report.get('n_benchmarks', 'n/a')}")
        print()
        print("  Top-3 leaderboard entries:")
        for entry in report.get("leaderboard", [])[:3]:
            print(f"    #{entry['rank']}  {entry['model_name']:<18}  "
                  f"avg={entry['avg_accuracy']:.1%}  "
                  f"params={entry.get('params', '?')}")

divider()
print("  Full pipeline complete.")
print("  To save reports to disk: set OUTPUT_DIR=outputs/ or pass output_dir= to ReportGenerator")
divider()
