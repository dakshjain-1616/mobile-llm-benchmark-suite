---
title: Mobile LLM Benchmark Suite
emoji: 📊
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
tags:
  - benchmark
  - evaluation
  - mobile-llm
  - llm
  - nlp
  - gsm8k
  - arc
  - mmlu
  - hellaswag
  - phi-4-mini
  - gemma
  - qwen
  - llama
  - smollm
---

# Mobile LLM Benchmark Suite

> *Made autonomously using [NEO](https://heyneo.so) — your autonomous AI Agent*

Rigorous evaluation harness for **1B–4B parameter models** — the class designed for mobile and edge deployment. Compare Phi-4-mini, Gemma-3, Qwen2.5, SmolLM3, and Llama-3.2 across 6 standard benchmarks with **Wilson 95% CI** and **Cohen's h effect sizes** for statistically valid comparisons. No GPU required.

---

## Live Demo

This Space hosts an interactive Gradio UI. Select models, choose benchmarks, set sample count — results appear with confidence intervals and pairwise significance tests.

---

## Models Evaluated

| Model | Parameters | Family |
|-------|-----------|--------|
| Phi-4-mini | 3.8B | Microsoft |
| Gemma-3-1B | 1.0B | Google |
| Gemma-3-4B | 4.0B | Google |
| Qwen2.5-1.5B | 1.5B | Alibaba |
| Qwen2.5-3B | 3.0B | Alibaba |
| SmolLM2-1.7B | 1.7B | HuggingFace |
| SmolLM3-3B | 3.0B | HuggingFace |
| Llama-3.2-1B | 1.0B | Meta |
| Llama-3.2-3B | 3.0B | Meta |

---

## Benchmarks

| Benchmark | Task Type | Metric |
|-----------|-----------|--------|
| GSM8K | Math reasoning | Exact match |
| ARC-Challenge | Science QA | Exact match |
| MMLU | General knowledge | Exact match |
| HellaSwag | Commonsense NLI | Exact match |
| TruthfulQA | Factual accuracy | Regex match |
| IFEval | Instruction following | Rule-based |

---

## Statistical Methodology

- **Wilson score interval** (95% CI) — correct for small sample sizes and extreme proportions
- **Cohen's h effect size** — quantifies practical significance of model differences
- **Two-sided z-test** — validates whether performance gaps are statistically significant

```python
from mobile_llm_benchmark.statistical import wilson_ci, cohens_h

ci_low, ci_high = wilson_ci(correct=72, total=100)
# → (0.624, 0.803) at 95% confidence

effect = cohens_h(p1=0.724, p2=0.618)
# → 0.23 (Small effect — likely real, worth noting)
```

---

## Install & Use Locally

```bash
git clone https://github.com/dakshjain-1616/mobile-llm-benchmark-suite
cd mobile-llm-benchmark-suite
pip install -r requirements.txt

# Run Gradio UI
python app.py

# CLI benchmark (mock mode, no API key)
python demo.py --mock
```

---

## Citation

```bibtex
@misc{mobile-llm-benchmark-2026,
  title   = {Mobile LLM Benchmark Suite},
  author  = {dakshjain-1616},
  year    = {2026},
  url     = {https://huggingface.co/spaces/daksh-neo/mobile-llm-benchmark-suite},
  note    = {Wilson 95% CI, Cohen's h, 6 benchmarks, 10 mobile-class models}
}
```
