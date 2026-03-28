#!/usr/bin/env python3
"""03_custom_config.py — Customise behaviour via env vars and config objects.

Demonstrates:
  - Overriding models via environment variables (MODEL_* overrides)
  - Using scenario presets from SCENARIO_PRESETS
  - Adjusting MOCK_SEED for reproducibility
  - Configuring output directory and log level at runtime

Run:
    python3 examples/03_custom_config.py
    # or with a custom seed:
    MOCK_SEED=123 python3 examples/03_custom_config.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")   # suppress info noise in example

from mobile_llm_benchmark import BenchmarkRunner, MODELS
from mobile_llm_benchmark.config import SCENARIO_PRESETS, MODEL_BY_NAME, BENCHMARK_IDS


# ── 1. Inspect available scenario presets ────────────────────────────────────
print("Available scenario presets:")
print(f"  {'Name':<14} {'Benchmarks':<40} {'Models':>5}  Samples")
print("  " + "-" * 72)
for key, preset in SCENARIO_PRESETS.items():
    benches = ", ".join(preset["benchmarks"])
    n_models = len(preset["models"])
    print(f"  {key:<14} {benches:<40} {n_models:>5}  {preset['n_samples']}")

# ── 2. Run the 'small_models' preset ─────────────────────────────────────────
print()
preset = SCENARIO_PRESETS["small_models"]
print(f"Running preset: '{preset['name']}' — {preset['description']}")

model_configs = [MODEL_BY_NAME[name] for name in preset["models"] if name in MODEL_BY_NAME]
benchmark_ids = preset["benchmarks"]
n_samples = preset["n_samples"]

runner = BenchmarkRunner(mock_mode=True)
results = runner.run(model_configs, benchmark_ids, n_samples=n_samples)

# ── 3. Show results grouped by benchmark ─────────────────────────────────────
print()
for bench_id in benchmark_ids:
    bench_results = [r for r in results if r.benchmark == bench_id]
    if not bench_results:
        continue
    print(f"  {bench_results[0].benchmark_name}:")
    for r in sorted(bench_results, key=lambda r: -r.accuracy):
        bar = "█" * int(r.accuracy * 25)
        print(f"    {r.model_name:<16} {r.accuracy:>6.1%}  {bar}")
    print()

# ── 4. Demonstrate MOCK_SEED reproducibility ─────────────────────────────────
seed = int(os.getenv("MOCK_SEED", "42"))
print(f"MOCK_SEED={seed}  (set MOCK_SEED=<n> for different reproducible results)")
sample_result = results[0]
print(f"  First result: {sample_result.model_name} / {sample_result.benchmark} = "
      f"{sample_result.accuracy:.1%}  n_correct={sample_result.n_correct}")

# ── 5. Configurable model IDs via env ─────────────────────────────────────────
print()
print("Model ID overrides (set these env vars to use different model endpoints):")
override_vars = [
    "MODEL_PHI4_MINI", "MODEL_GEMMA3_1B", "MODEL_GEMMA3_4B",
    "MODEL_QWEN25_15B", "MODEL_QWEN25_3B", "MODEL_SMOLLM2",
]
for var in override_vars:
    val = os.getenv(var, "(default)")
    print(f"  {var:<22} = {val}")
