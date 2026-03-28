"""BenchmarkRunner: orchestrates evaluation of models across benchmarks."""

from __future__ import annotations

import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

import numpy as np

from .config import MODEL_CAPABILITIES, DEFAULT_CAPABILITY
from .data_loaders import DataLoader
from .model_client import ModelClient
from .scorers import get_scorer
from .statistical import wilson_ci

logger = logging.getLogger(__name__)

_MOCK_SEED = int(os.getenv("MOCK_SEED", "42"))


@dataclass
class BenchmarkResult:
    """Result for a single (model, benchmark) pair."""

    model_name: str
    model_id: str
    benchmark: str
    benchmark_name: str
    accuracy: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    n_correct: int
    duration_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mock: bool = False

    # Token usage (populated in real mode; 0 in mock mode)
    tokens_prompt: int = 0
    tokens_completion: int = 0

    # Per-question latency (ms) — populated in real mode
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Category breakdown (e.g. MMLU subject → accuracy) — may be empty
    category_scores: dict[str, float] = field(default_factory=dict)

    # Convenience
    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "model_id": self.model_id,
            "benchmark": self.benchmark,
            "benchmark_name": self.benchmark_name,
            "accuracy": round(self.accuracy, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "n_samples": self.n_samples,
            "n_correct": self.n_correct,
            "duration_s": round(self.duration_s, 2),
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "tokens_total": self.tokens_total,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "timestamp": self.timestamp,
            "mock": self.mock,
        }


class BenchmarkRunner:
    """Runs benchmarks against models, either via API calls or in mock mode."""

    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        self.data_loader = DataLoader(mock_mode=mock_mode, seed=_MOCK_SEED)
        self.client = ModelClient() if not mock_mode else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        model_configs: list[dict],
        benchmark_ids: list[str],
        n_samples: int = 50,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[BenchmarkResult]:
        """Run all (model, benchmark) pairs and return results."""
        results: list[BenchmarkResult] = []
        total = len(model_configs) * len(benchmark_ids)
        done = 0

        for model_cfg in model_configs:
            for bench_id in benchmark_ids:
                if progress_callback:
                    pct = done / total
                    progress_callback(
                        f"Evaluating {model_cfg['name']} on {bench_id} "
                        f"({done+1}/{total})",
                        pct,
                    )
                try:
                    result = self.run_single(model_cfg, bench_id, n_samples)
                    results.append(result)
                    tok_info = (
                        f" tokens={result.tokens_total}"
                        if result.tokens_total > 0
                        else ""
                    )
                    logger.info(
                        "✓ %s / %s: accuracy=%.1f%% [%.1f%%–%.1f%%] n=%d%s",
                        model_cfg["name"],
                        bench_id,
                        result.accuracy * 100,
                        result.ci_lower * 100,
                        result.ci_upper * 100,
                        result.n_samples,
                        tok_info,
                    )
                except Exception as exc:
                    logger.error(
                        "✗ %s / %s failed: %s", model_cfg["name"], bench_id, exc
                    )
                done += 1

        if progress_callback:
            progress_callback("Done", 1.0)
        return results

    def run_single(
        self, model_cfg: dict, bench_id: str, n_samples: int
    ) -> BenchmarkResult:
        """Evaluate one model on one benchmark."""
        from .config import BENCHMARK_BY_ID

        bench_meta = BENCHMARK_BY_ID[bench_id]
        bench_name = bench_meta["name"]

        if self.mock_mode:
            return self._mock_run(model_cfg, bench_id, bench_name, n_samples)
        return self._real_run(model_cfg, bench_id, bench_name, n_samples)

    # ------------------------------------------------------------------
    # Real evaluation
    # ------------------------------------------------------------------

    def _real_run(
        self, model_cfg: dict, bench_id: str, bench_name: str, n_samples: int
    ) -> BenchmarkResult:
        scorer = get_scorer(bench_id)
        questions = self.data_loader.load(bench_id, n_samples)

        t0 = time.perf_counter()
        n_correct = 0
        n_evaluated = 0
        total_tokens_prompt = 0
        total_tokens_completion = 0
        latencies_ms: list[float] = []

        # Per-category tracking (e.g. MMLU subjects)
        cat_correct: dict[str, int] = defaultdict(int)
        cat_total: dict[str, int] = defaultdict(int)

        for q in questions:
            try:
                prompt = scorer.build_prompt(q)
                result = self.client.generate_full(
                    model_id=model_cfg["id"],
                    prompt=prompt,
                    provider=model_cfg.get("provider", "openrouter"),
                    max_tokens=scorer.max_tokens(),
                    temperature=scorer.temperature(),
                )
                correct = scorer.score(q, result.text)
                if correct:
                    n_correct += 1
                n_evaluated += 1
                total_tokens_prompt += result.tokens_prompt
                total_tokens_completion += result.tokens_completion
                latencies_ms.append(result.latency_ms)

                # Category tracking
                category = q.get("subject") or q.get("category")
                if category:
                    cat_total[category] += 1
                    if correct:
                        cat_correct[category] += 1

            except Exception as exc:
                logger.warning("Skipping question due to error: %s", exc)

        duration = time.perf_counter() - t0

        if n_evaluated == 0:
            raise RuntimeError(f"No questions evaluated for {model_cfg['name']} / {bench_id}")

        avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        category_scores = {
            cat: cat_correct[cat] / cat_total[cat]
            for cat in cat_total
            if cat_total[cat] > 0
        }

        ci_lower, accuracy, ci_upper = wilson_ci(n_correct, n_evaluated)
        return BenchmarkResult(
            model_name=model_cfg["name"],
            model_id=model_cfg["id"],
            benchmark=bench_id,
            benchmark_name=bench_name,
            accuracy=accuracy,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n_evaluated,
            n_correct=n_correct,
            duration_s=duration,
            tokens_prompt=total_tokens_prompt,
            tokens_completion=total_tokens_completion,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            category_scores=category_scores,
            mock=False,
        )

    # ------------------------------------------------------------------
    # Mock evaluation
    # ------------------------------------------------------------------

    def _mock_run(
        self, model_cfg: dict, bench_id: str, bench_name: str, n_samples: int
    ) -> BenchmarkResult:
        """Simulate benchmark results using stored capability scores + noise."""
        rng = random.Random(_MOCK_SEED + hash(model_cfg["name"] + bench_id) % 10000)

        cap = MODEL_CAPABILITIES.get(model_cfg["name"], DEFAULT_CAPABILITY)
        base_p = cap.get(bench_id, DEFAULT_CAPABILITY.get(bench_id, 0.45))

        # Add small Gaussian noise (σ=0.02) for realism
        noise = rng.gauss(0, 0.02)
        true_p = min(0.95, max(0.10, base_p + noise))

        # Sample individual outcomes
        outcomes = [1 if rng.random() < true_p else 0 for _ in range(n_samples)]
        n_correct = sum(outcomes)

        # Simulate ~0.5s per question (compressed for mock)
        duration = n_samples * 0.01

        # Mock token usage — approximate counts based on typical prompt/response sizes
        mock_tokens_prompt = n_samples * rng.randint(40, 80)
        mock_tokens_completion = n_samples * rng.randint(8, 32)
        mock_avg_lat = rng.gauss(320, 60)  # ~320ms avg simulated latency
        mock_p95_lat = mock_avg_lat + rng.gauss(120, 30)

        ci_lower, accuracy, ci_upper = wilson_ci(n_correct, n_samples)
        return BenchmarkResult(
            model_name=model_cfg["name"],
            model_id=model_cfg["id"],
            benchmark=bench_id,
            benchmark_name=bench_name,
            accuracy=accuracy,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n_samples,
            n_correct=n_correct,
            duration_s=duration,
            tokens_prompt=mock_tokens_prompt,
            tokens_completion=mock_tokens_completion,
            avg_latency_ms=max(50.0, mock_avg_lat),
            p95_latency_ms=max(100.0, mock_p95_lat),
            mock=True,
        )

    # ------------------------------------------------------------------
    # Generator variant (for Gradio streaming)
    # ------------------------------------------------------------------

    def run_stream(
        self,
        model_configs: list[dict],
        benchmark_ids: list[str],
        n_samples: int = 50,
    ) -> Iterator[tuple[list[BenchmarkResult], str]]:
        """Run benchmarks and yield (results_so_far, log_line) after each pair."""
        results: list[BenchmarkResult] = []
        total = len(model_configs) * len(benchmark_ids)
        done = 0

        for model_cfg in model_configs:
            for bench_id in benchmark_ids:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                log = (
                    f"[{ts}] [{done+1}/{total}] {model_cfg['name']} → {bench_id} "
                    f"({'mock' if self.mock_mode else 'live'})..."
                )
                yield results, log

                try:
                    result = self.run_single(model_cfg, bench_id, n_samples)
                    results.append(result)
                    tok_info = f"  tokens={result.tokens_total}" if result.tokens_total > 0 else ""
                    lat_info = f"  p95={result.p95_latency_ms:.0f}ms" if result.p95_latency_ms > 0 else ""
                    log = (
                        f"  ✓ {model_cfg['name']} / {bench_id}: "
                        f"{result.accuracy:.1%} [{result.ci_lower:.1%}–{result.ci_upper:.1%}]"
                        f"{tok_info}{lat_info}"
                    )
                except Exception as exc:
                    log = f"  ✗ {model_cfg['name']} / {bench_id}: ERROR — {exc}"

                done += 1
                yield results, log

        # Summary stats
        total_tokens = sum(r.tokens_total for r in results)
        token_summary = f"  Total tokens used: {total_tokens:,}" if total_tokens > 0 else ""
        yield results, f"\n✓ Benchmark complete — {len(results)}/{total} pairs evaluated.{token_summary}"
