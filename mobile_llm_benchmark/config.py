"""Central configuration: models, benchmarks, capabilities, scenario presets."""

import os

# ---------------------------------------------------------------------------
# Model registry
# Each model has: id (API string), name (display), params, provider
# OpenRouter IDs use lowercase provider/model-name convention.
# HF Inference API IDs use the exact HuggingFace repo slug.
# ---------------------------------------------------------------------------

MODELS = [
    {
        "id": os.getenv("MODEL_PHI4_MINI", "microsoft/phi-4-mini-instruct"),
        "name": "Phi-4-Mini",
        "params": "3.8B",
        "provider": "openrouter",
    },
    {
        "id": os.getenv("MODEL_GEMMA3_1B", "google/gemma-3-1b-it"),
        "name": "Gemma-3-1B",
        "params": "1B",
        "provider": "openrouter",
    },
    {
        "id": os.getenv("MODEL_GEMMA3_4B", "google/gemma-3-4b-it"),
        "name": "Gemma-3-4B",
        "params": "4B",
        "provider": "openrouter",
    },
    {
        "id": os.getenv("MODEL_QWEN25_15B", "Qwen/Qwen2.5-1.5B-Instruct"),
        "name": "Qwen2.5-1.5B",
        "params": "1.5B",
        "provider": "hf",
    },
    {
        "id": os.getenv("MODEL_QWEN25_3B", "Qwen/Qwen2.5-3B-Instruct"),
        "name": "Qwen2.5-3B",
        "params": "3B",
        "provider": "hf",
    },
    {
        "id": os.getenv("MODEL_SMOLLM2", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
        "name": "SmolLM2-1.7B",
        "params": "1.7B",
        "provider": "hf",
    },
    # Qwen3-0.6B — latest ultra-compact Qwen3 model (HF trending)
    {
        "id": os.getenv("MODEL_QWEN3_06B", "Qwen/Qwen3-0.6B"),
        "name": "Qwen3-0.6B",
        "params": "0.6B",
        "provider": "hf",
    },
    {
        "id": os.getenv("MODEL_LLAMA32_1B", "meta-llama/llama-3.2-1b-instruct"),
        "name": "Llama-3.2-1B",
        "params": "1B",
        "provider": "openrouter",
    },
    {
        "id": os.getenv("MODEL_LLAMA32_3B", "meta-llama/Llama-3.2-3B-Instruct"),
        "name": "Llama-3.2-3B",
        "params": "3B",
        "provider": "hf",
    },
    {
        "id": os.getenv("MODEL_QWEN3_17B", "Qwen/Qwen3-1.7B"),
        "name": "Qwen3-1.7B",
        "params": "1.7B",
        "provider": "hf",
    },
]

# Display names → model dict (for quick lookup)
MODEL_BY_NAME = {m["name"]: m for m in MODELS}
MODEL_BY_ID = {m["id"]: m for m in MODELS}

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARKS = [
    {
        "id": "gsm8k",
        "name": "GSM8K",
        "description": "Grade-school math word problems",
        "dataset": "gsm8k",
        "dataset_config": "main",
        "split": "test",
        "metric": "exact_match",
        "n_shots": 0,
    },
    {
        "id": "arc_challenge",
        "name": "ARC-Challenge",
        "description": "AI2 Reasoning Challenge — hard science questions",
        "dataset": "ai2_arc",
        "dataset_config": "ARC-Challenge",
        "split": "test",
        "metric": "multiple_choice",
        "n_shots": 0,
    },
    {
        "id": "mmlu",
        "name": "MMLU",
        "description": "Massive Multitask Language Understanding",
        "dataset": "cais/mmlu",
        "dataset_config": "all",
        "split": "test",
        "metric": "multiple_choice",
        "n_shots": 0,
    },
    {
        "id": "hellaswag",
        "name": "HellaSwag",
        "description": "Commonsense NLI — pick best sentence completion",
        "dataset": "hellaswag",
        "dataset_config": None,
        "split": "validation",
        "metric": "multiple_choice",
        "n_shots": 0,
    },
    {
        "id": "truthfulqa",
        "name": "TruthfulQA",
        "description": "Truthfulness evaluation — MC format",
        "dataset": "truthful_qa",
        "dataset_config": "multiple_choice",
        "split": "validation",
        "metric": "multiple_choice",
        "n_shots": 0,
    },
    {
        "id": "ifeval",
        "name": "IFEval",
        "description": "Instruction Following Evaluation",
        "dataset": "google/IFEval",
        "dataset_config": None,
        "split": "train",
        "metric": "instruction_following",
        "n_shots": 0,
    },
]

BENCHMARK_BY_ID = {b["id"]: b for b in BENCHMARKS}
BENCHMARK_NAMES = [b["name"] for b in BENCHMARKS]
BENCHMARK_IDS = [b["id"] for b in BENCHMARKS]

# ---------------------------------------------------------------------------
# Scenario presets — curated benchmark/sample combinations for common use-cases
# ---------------------------------------------------------------------------

SCENARIO_PRESETS: dict[str, dict] = {
    "quick": {
        "name": "Quick Test",
        "description": "Fast sanity check — 2 benchmarks, 20 samples (~2 min in mock)",
        "emoji": "⚡",
        "benchmarks": ["arc_challenge", "mmlu"],
        "n_samples": 20,
        "models": ["Llama-3.2-1B", "SmolLM2-1.7B", "Qwen2.5-3B"],
    },
    "math": {
        "name": "Math & Reasoning",
        "description": "Math and scientific reasoning benchmarks",
        "emoji": "🔢",
        "benchmarks": ["gsm8k", "arc_challenge", "mmlu"],
        "n_samples": 50,
        "models": ["Phi-4-Mini", "Gemma-3-4B", "Qwen2.5-3B", "Llama-3.2-3B"],
    },
    "language": {
        "name": "Language Understanding",
        "description": "NLI, truthfulness, instruction following",
        "emoji": "📚",
        "benchmarks": ["hellaswag", "truthfulqa", "ifeval"],
        "n_samples": 50,
        "models": ["Phi-4-Mini", "Gemma-3-4B", "Qwen3-1.7B", "Llama-3.2-3B"],
    },
    "small_models": {
        "name": "Tiny Models (<1B)",
        "description": "Compare the smallest available models",
        "emoji": "🔬",
        "benchmarks": ["arc_challenge", "mmlu", "hellaswag"],
        "n_samples": 30,
        "models": ["Qwen3-0.6B", "Gemma-3-1B", "Llama-3.2-1B"],
    },
    "full": {
        "name": "Full Suite",
        "description": "All 6 benchmarks, all 10 models, 100 samples",
        "emoji": "🏆",
        "benchmarks": ["gsm8k", "arc_challenge", "mmlu", "hellaswag", "truthfulqa", "ifeval"],
        "n_samples": 100,
        "models": [m["name"] for m in MODELS],
    },
}

# ---------------------------------------------------------------------------
# Model capability scores for mock mode (0-1, higher = stronger)
# Based on published benchmarks and model size scaling laws
# ---------------------------------------------------------------------------

MODEL_CAPABILITIES: dict[str, dict[str, float]] = {
    "Phi-4-Mini": {
        "gsm8k": 0.72,
        "arc_challenge": 0.68,
        "mmlu": 0.65,
        "hellaswag": 0.78,
        "truthfulqa": 0.55,
        "ifeval": 0.62,
    },
    "Gemma-3-4B": {
        "gsm8k": 0.65,
        "arc_challenge": 0.64,
        "mmlu": 0.60,
        "hellaswag": 0.75,
        "truthfulqa": 0.52,
        "ifeval": 0.58,
    },
    "Qwen2.5-3B": {
        "gsm8k": 0.60,
        "arc_challenge": 0.60,
        "mmlu": 0.57,
        "hellaswag": 0.72,
        "truthfulqa": 0.50,
        "ifeval": 0.55,
    },
    "Llama-3.2-3B": {
        "gsm8k": 0.55,
        "arc_challenge": 0.58,
        "mmlu": 0.53,
        "hellaswag": 0.70,
        "truthfulqa": 0.48,
        "ifeval": 0.50,
    },
    "Gemma-3-1B": {
        "gsm8k": 0.38,
        "arc_challenge": 0.48,
        "mmlu": 0.42,
        "hellaswag": 0.62,
        "truthfulqa": 0.40,
        "ifeval": 0.38,
    },
    "Qwen2.5-1.5B": {
        "gsm8k": 0.42,
        "arc_challenge": 0.50,
        "mmlu": 0.46,
        "hellaswag": 0.64,
        "truthfulqa": 0.42,
        "ifeval": 0.40,
    },
    "SmolLM2-1.7B": {
        "gsm8k": 0.35,
        "arc_challenge": 0.46,
        "mmlu": 0.40,
        "hellaswag": 0.60,
        "truthfulqa": 0.38,
        "ifeval": 0.35,
    },
    "Qwen3-0.6B": {
        "gsm8k": 0.28,
        "arc_challenge": 0.40,
        "mmlu": 0.35,
        "hellaswag": 0.55,
        "truthfulqa": 0.35,
        "ifeval": 0.30,
    },
    "Llama-3.2-1B": {
        "gsm8k": 0.30,
        "arc_challenge": 0.42,
        "mmlu": 0.38,
        "hellaswag": 0.58,
        "truthfulqa": 0.36,
        "ifeval": 0.32,
    },
    "Qwen3-1.7B": {
        "gsm8k": 0.48,
        "arc_challenge": 0.54,
        "mmlu": 0.50,
        "hellaswag": 0.68,
        "truthfulqa": 0.46,
        "ifeval": 0.44,
    },
}

# Default capability for any model not in the table
DEFAULT_CAPABILITY: dict[str, float] = {
    "gsm8k": 0.40,
    "arc_challenge": 0.50,
    "mmlu": 0.45,
    "hellaswag": 0.62,
    "truthfulqa": 0.40,
    "ifeval": 0.40,
}
