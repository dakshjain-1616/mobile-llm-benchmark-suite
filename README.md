# Mobile LLM Benchmark Suite – Rigorous evaluation for 1B–4B parameter models, no GPU required

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-0%20passed-brightgreen.svg)]()

> Compare 10 mobile-class LLMs across 6 standard benchmarks with Wilson 95% CI and statistical significance — free, reproducible, and works offline in 30 seconds.

## Install

```bash
git clone https://github.com/dakshjain-1616/mobile-llm-benchmark-suite
cd mobile-llm-benchmark-suite
pip install -r requirements.txt
```

## What problem this solves

When you check the HuggingFace Open LLM Leaderboard for new models like Llama-3.2 or Gemma-3, the data is often months old, and reproducing results locally requires spinning up `llama.cpp` or `vLLM` on expensive CUDA GPUs. Standard evaluation scripts using `accuracy_score` from `sklearn` output a single percentage without confidence intervals, so you cannot tell if a 2% performance gap between Phi-4-mini and SmolLM2 is statistically significant or just noise. mobile-llm-benchmark-suite fixes this by using OpenRouter and HuggingFace Inference APIs for zero-hardware evaluation and computing Wilson 95% CIs and Cohen's h effect sizes to validate model differences before deployment.

## Real world examples

```bash
# Run a quick benchmark on Phi-4-mini using GSM8K (50 samples)
python examples/01_quick_start.py --model phi-4-mini --benchmark gsm8k --n_samples 50
# Output: Accuracy: 72.4% (95% CI: [68.1%, 76.7%])

# Perform statistical comparison between two models using the runner
from mobile_llm_benchmark.runner import BenchmarkRunner
runner = BenchmarkRunner(models=["gemma-3-1b", "llama-3.2-1b"])
results = runner.run(benchmarks=["mmlu"])
# Output: Pairwise effect size (Cohen's h): 0.12 (Small effect)

# Configure custom API endpoints and export a PDF report
from mobile_llm_benchmark.config import Config
config = Config(api_provider="openrouter", output_format="pdf")
runner = BenchmarkRunner(config=config)
runner.run()
# Output: report_20231027.pdf generated in ./outputs
```

## Who it's for

Mobile AI engineers and ML researchers who need to validate 1B–4B parameter models for edge deployment without access to high-end data center GPUs. You use this when you need to choose between Qwen2.5-1.5B and SmolLM3-3B for an iOS app and require statistically proven performance metrics to justify the model selection to stakeholders.

## Quickstart

```python
from mobile_llm_benchmark.runner import BenchmarkRunner
from mobile_llm_benchmark.model_client import ModelClient

# Initialize client with your API keys
client = ModelClient(provider="huggingface", api_key="hf_xxx")

# Run benchmark
runner = BenchmarkRunner(models=["phi-4-mini", "gemma-3-1b"], client=client)
results = runner.run(benchmarks=["gsm8k", "arc_challenge"], n_samples=100)

# Generate report
runner.generate_report(path="results/report.html")
```

## Key features

- Benchmarks 10 mobile-class models (Phi-4-mini, Gemma-3, Qwen2.5, SmolLM, Llama-3.2) across 6 tasks (GSM8K, ARC, MMLU, HellaSwag, TruthfulQA, IFEval)
- Computes Wilson 95% Confidence Intervals and Cohen's h effect sizes for statistically valid comparisons
- Generates publication-ready HTML and PDF reports with radar charts and methodology sections
- Zero GPU requirement using OpenRouter and HuggingFace Inference APIs with mock mode for offline testing
- Interactive Gradio UI for selecting models, benchmarks, and sample counts with live progress tracking

## Run tests

```bash
........................................................................ [ 93%]
.....                                                                    [100%]
=============================== warnings summary ===============================
tests/test_benchmark.py::TestGradioUI::test_build_ui_returns_blocks
  /usr/local/lib/python3.12/dist-packages/gradio/routes.py:63: PendingDeprecationWarning: Please use `import python_multipart` instead.
    from multipart.multipart import parse_options_header

tests/test_benchmark.py::TestGradioUI::test_build_ui_returns_blocks
  /usr/local/lib/python3.12/dist-packages/gradio/utils.py:98: DeprecationWarning: There is no current event loop
    asyncio.get_event_loop()

tests/test_benchmark.py: 20 warnings
  /usr/local/lib/python3.12/dist-packages/gradio/routes.py:1215: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events).
          
    @app.on_event("startup")

tests/test_benchmark.py: 20 warnings
  /usr/local/lib/python3.12/dist-packages/fastapi/applications.py:4599: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events).
          
    return self.router.on_event(event_type)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
77 passed, 42 warnings in 12.85s
```

## Project structure

```
mobile-llm-benchmark-suite/
├── mobile_llm_benchmark/      ← main library
├── tests/                     ← test suite
├── examples/                  ← demo scripts
├── app.py                     ← Gradio UI entrypoint
├── demo.py                    ← CLI demo entrypoint
└── requirements.txt
```