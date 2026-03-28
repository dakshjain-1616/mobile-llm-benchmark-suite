# Examples

Runnable scripts demonstrating the Mobile LLM Benchmark Suite.
All examples work in **mock mode** — no API key required.

```bash
python3 examples/01_quick_start.py
```

## Scripts

| Script | What it demonstrates |
|--------|----------------------|
| [01_quick_start.py](01_quick_start.py) | Minimal working example: 2 models × 2 benchmarks, prints results table and per-model average |
| [02_advanced_usage.py](02_advanced_usage.py) | Streaming evaluation with `run_stream()`, Wilson vs bootstrap CI comparison, Cohen's h effect sizes, and pairwise z-test significance |
| [03_custom_config.py](03_custom_config.py) | Scenario presets, `MOCK_SEED` reproducibility, runtime env-var overrides (`MODEL_*`, `LOG_LEVEL`) |
| [04_full_pipeline.py](04_full_pipeline.py) | End-to-end workflow: configure → stream-evaluate → leaderboard → pairwise stats → HTML/CSV/JSON report generation with JSON excerpt |

## Running from any directory

Each script adds the project root to `sys.path` automatically, so you can run them from anywhere:

```bash
# from project root
python3 examples/01_quick_start.py

# from examples/ directory
cd examples && python3 01_quick_start.py

# with a custom seed
MOCK_SEED=123 python3 examples/03_custom_config.py
```

## Using a real API

Set `OPENROUTER_API_KEY` or `HUGGINGFACE_TOKEN` and remove the `MOCK_MODE=true` override at the top of any script:

```bash
OPENROUTER_API_KEY=sk-... python3 examples/04_full_pipeline.py
```

See `.env.example` in the project root for all available configuration options.
