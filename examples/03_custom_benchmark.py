"""
Example 03: Custom Benchmark Run with Report Generation
=======================================================
Demonstrates:
  - Selecting specific models and benchmarks programmatically
  - Streaming progress updates during evaluation
  - Generating full HTML/CSV/JSON reports
  - Loading results back from CSV for analysis

Run:
    python3 examples/03_custom_benchmark.py
"""

import os
import sys
import json
import csv
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MOCK_MODE", "true")

from mobile_llm_benchmark import BenchmarkRunner
from mobile_llm_benchmark.report_generator import ReportGenerator


def progress_bar(pct: float, width: int = 30) -> str:
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def main():
    print("=" * 65)
    print("Mobile LLM Benchmark Suite — Custom Run + Report Generation")
    print("=" * 65)

    # ── Configuration ────────────────────────────────────────────────────────
    MODEL_CONFIGS = [
        {
            "name": "Phi-4-Mini",
            "id": "microsoft/phi-4-mini-instruct",
            "params": "3.8B",
            "provider": "openrouter",
        },
        {
            "name": "Gemma-3-4B",
            "id": "google/gemma-3-4b-it",
            "params": "4B",
            "provider": "openrouter",
        },
        {
            "name": "Qwen2.5-3B",
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "params": "3B",
            "provider": "hf",
        },
    ]
    BENCHMARK_IDS = ["gsm8k", "arc_challenge", "mmlu"]
    N_SAMPLES = 30

    print()
    print(f"Models     : {', '.join(m['name'] for m in MODEL_CONFIGS)}")
    print(f"Benchmarks : {', '.join(BENCHMARK_IDS)}")
    print(f"Samples    : {N_SAMPLES} per benchmark")
    print(f"Mode       : mock (offline)")
    print()

    # ── Streaming run with progress ──────────────────────────────────────────
    runner = BenchmarkRunner(mock_mode=True)

    results = []
    print("Running evaluation …")
    total = len(MODEL_CONFIGS) * len(BENCHMARK_IDS)
    done = 0

    for partial_results, log_line in runner.run_stream(
        model_configs=MODEL_CONFIGS,
        benchmark_ids=BENCHMARK_IDS,
        n_samples=N_SAMPLES,
    ):
        results = partial_results
        done = len(results)
        pct = done / total
        bar = progress_bar(pct)
        # Parse model/bench from the log line if available
        print(f"\r  [{bar}] {done}/{total}  {log_line[:45]:<45}", end="", flush=True)

    print(f"\r  [{'█'*30}] {total}/{total}  Complete!               ")
    print()

    # ── Results table ────────────────────────────────────────────────────────
    print("Results:")
    col_w = [14, 14, 10, 24, 8]
    headers = ["Model", "Benchmark", "Accuracy", "95% CI", "Samples"]
    row_fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    sep     = "  " + "  ".join("-" * w for w in col_w)

    print(row_fmt.format(*headers))
    print(sep)
    for r in sorted(results, key=lambda r: (-r.accuracy, r.model_name, r.benchmark)):
        ci_str = f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        print(row_fmt.format(
            r.model_name, r.benchmark_name,
            f"{r.accuracy:.1%}", ci_str, str(r.n_samples),
        ))

    # ── Leaderboard summary ──────────────────────────────────────────────────
    print()
    print("Leaderboard (avg across all benchmarks):")
    model_scores: dict[str, list[float]] = {}
    for r in results:
        model_scores.setdefault(r.model_name, []).append(r.accuracy)
    ranked = sorted(model_scores.items(), key=lambda kv: -sum(kv[1])/len(kv[1]))
    for rank, (model, scores) in enumerate(ranked, 1):
        avg = sum(scores) / len(scores)
        bar = progress_bar(avg, width=25)
        print(f"  #{rank}  {model:<14}  {avg:.1%}  {bar}")

    # ── Report generation ────────────────────────────────────────────────────
    print()
    print("Generating reports …")
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator(output_dir=tmpdir)
        artifacts = gen.generate_all(results)

        print("  Generated artifacts:")
        for name, path in artifacts.items():
            if path and os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"    ✓  {name:<22}  {size_kb:>7.1f} KB  →  {os.path.basename(path)}")

        # Peek at JSON
        json_path = artifacts.get("json")
        if json_path and os.path.exists(json_path):
            print()
            print("  JSON leaderboard (first entry):")
            with open(json_path) as f:
                data = json.load(f)
            leader = data["leaderboard"][0]
            print(f"    Rank 1: {leader['model_name']}  avg={leader['avg_accuracy']:.1%}")

    print()
    print("Done! To save reports to disk, pass output_dir='outputs/' to ReportGenerator.")


if __name__ == "__main__":
    main()
