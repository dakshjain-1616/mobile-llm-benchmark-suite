"""pytest suite for Mobile LLM Benchmark Suite.

All tests run in mock mode — no API keys required.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force mock mode for all tests
os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("OUTPUT_DIR", str(Path(tempfile.mkdtemp()) / "test_outputs"))


# =============================================================================
# Test 1: Wilson CI correctness
# =============================================================================

class TestWilsonCI:
    def test_ci_ordering_standard(self):
        """ci_lower < accuracy < ci_upper for a typical case."""
        from mobile_llm_benchmark.statistical import wilson_ci
        ci_low, acc, ci_high = wilson_ci(35, 50)
        assert 0 <= ci_low < acc < ci_high <= 1, (
            f"Expected ci_lower < accuracy < ci_upper, got {ci_low:.4f} < {acc:.4f} < {ci_high:.4f}"
        )

    def test_ci_95_percent(self):
        """95% CI is narrower than 99% CI."""
        from mobile_llm_benchmark.statistical import wilson_ci
        lo95, acc, hi95 = wilson_ci(30, 50, confidence=0.95)
        lo99, _, hi99 = wilson_ci(30, 50, confidence=0.99)
        assert hi95 - lo95 < hi99 - lo99

    def test_ci_width_shrinks_with_n(self):
        """Larger n → narrower confidence interval."""
        from mobile_llm_benchmark.statistical import wilson_ci
        _, _, hi50 = wilson_ci(25, 50)
        lo50, _, _ = wilson_ci(25, 50)
        width50 = hi50 - lo50
        lo200, _, hi200 = wilson_ci(100, 200)
        width200 = hi200 - lo200
        assert width200 < width50

    def test_zero_samples(self):
        from mobile_llm_benchmark.statistical import wilson_ci
        ci_low, acc, ci_high = wilson_ci(0, 0)
        assert ci_low == 0.0 and acc == 0.0 and ci_high == 0.0

    def test_all_correct(self):
        """All correct: accuracy = 1, CI upper should be 1."""
        from mobile_llm_benchmark.statistical import wilson_ci
        ci_low, acc, ci_high = wilson_ci(50, 50)
        assert acc == 1.0
        assert ci_low < 1.0  # CI still has uncertainty
        assert ci_high == 1.0

    def test_none_correct(self):
        """None correct: accuracy = 0, CI lower = 0."""
        from mobile_llm_benchmark.statistical import wilson_ci
        ci_low, acc, ci_high = wilson_ci(0, 50)
        assert acc == 0.0
        assert ci_low == 0.0
        assert ci_high > 0.0  # CI extends up

    def test_bulk_ci_ordering(self):
        """ci_lower < accuracy < ci_upper for all non-degenerate results."""
        from mobile_llm_benchmark.statistical import wilson_ci
        for n_correct in range(1, 50):  # skip 0 and 50 (degenerate)
            lo, acc, hi = wilson_ci(n_correct, 50)
            assert lo < acc < hi, f"CI ordering failed for n_correct={n_correct}"

    def test_cohens_h(self):
        from mobile_llm_benchmark.statistical import cohens_h, effect_size_label
        h = cohens_h(0.5, 0.7)
        assert 0 < h < 1
        assert effect_size_label(0.1) == "negligible"
        assert effect_size_label(0.3) == "small"
        assert effect_size_label(0.6) == "medium"
        assert effect_size_label(0.9) == "large"


# =============================================================================
# Test 1b: Bootstrap CI (new)
# =============================================================================

class TestBootstrapCI:
    def test_ci_ordering(self):
        from mobile_llm_benchmark.statistical import bootstrap_ci
        outcomes = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 5
        lo, acc, hi = bootstrap_ci(outcomes)
        assert 0 <= lo <= acc <= hi <= 1

    def test_empty_outcomes(self):
        from mobile_llm_benchmark.statistical import bootstrap_ci
        lo, acc, hi = bootstrap_ci([])
        assert lo == acc == hi == 0.0

    def test_deterministic_with_seed(self):
        from mobile_llm_benchmark.statistical import bootstrap_ci
        outcomes = [1, 0, 1, 1, 0] * 10
        lo1, acc1, hi1 = bootstrap_ci(outcomes, seed=42)
        lo2, acc2, hi2 = bootstrap_ci(outcomes, seed=42)
        assert lo1 == lo2 and acc1 == acc2 and hi1 == hi2

    def test_matches_accuracy(self):
        from mobile_llm_benchmark.statistical import bootstrap_ci
        outcomes = [1] * 30 + [0] * 20
        _, acc, _ = bootstrap_ci(outcomes)
        assert abs(acc - 0.6) < 1e-9

    def test_wider_ci_than_wilson_for_small_n(self):
        """Bootstrap CI should be non-degenerate for small n."""
        from mobile_llm_benchmark.statistical import bootstrap_ci
        outcomes = [1, 0, 1, 0, 1]
        lo, acc, hi = bootstrap_ci(outcomes, n_bootstrap=500)
        assert hi - lo > 0


# =============================================================================
# Test 1c: Pairwise significance (new)
# =============================================================================

class TestPairwiseSignificance:
    def test_z_test_proportions_identical(self):
        """Two identical proportions → p-value near 1, not significant."""
        from mobile_llm_benchmark.statistical import z_test_proportions
        z, p = z_test_proportions(30, 50, 30, 50)
        assert p > 0.5

    def test_z_test_proportions_different(self):
        """Very different proportions → significant."""
        from mobile_llm_benchmark.statistical import z_test_proportions
        z, p = z_test_proportions(5, 50, 45, 50)
        assert p < 0.05

    def test_z_test_zero_n(self):
        from mobile_llm_benchmark.statistical import z_test_proportions
        z, p = z_test_proportions(0, 0, 25, 50)
        assert p == 1.0

    def test_pairwise_significance_returns_pairs(self):
        from mobile_llm_benchmark.statistical import pairwise_significance
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS
        runner = BenchmarkRunner(mock_mode=True)
        results = runner.run(MODELS[:3], ["arc_challenge"], n_samples=30)
        pairs = pairwise_significance(results, "arc_challenge")
        # 3 models → 3 pairs
        assert len(pairs) == 3
        for pair in pairs:
            assert "model_a" in pair
            assert "model_b" in pair
            assert "p_value" in pair
            assert "significant" in pair
            assert isinstance(pair["significant"], bool)
            assert 0 <= pair["p_value"] <= 1


# =============================================================================
# Test 2: Data loaders (mock mode)
# =============================================================================

class TestDataLoaders:
    def test_all_benchmarks_load(self):
        from mobile_llm_benchmark.data_loaders import DataLoader
        loader = DataLoader(mock_mode=True)
        bench_ids = ["gsm8k", "arc_challenge", "mmlu", "hellaswag", "truthfulqa", "ifeval"]
        for bid in bench_ids:
            questions = loader.load(bid, n_samples=5)
            assert len(questions) == 5, f"Expected 5 questions for {bid}, got {len(questions)}"

    def test_question_structure_gsm8k(self):
        from mobile_llm_benchmark.data_loaders import DataLoader
        loader = DataLoader(mock_mode=True)
        questions = loader.load("gsm8k", n_samples=3)
        for q in questions:
            assert "question" in q
            assert "answer" in q

    def test_question_structure_arc(self):
        from mobile_llm_benchmark.data_loaders import DataLoader
        loader = DataLoader(mock_mode=True)
        questions = loader.load("arc_challenge", n_samples=3)
        for q in questions:
            assert "question" in q
            assert "choices" in q
            assert "answer" in q
            assert q["answer"] in ["A", "B", "C", "D"]

    def test_cycling_behavior(self):
        """Can request more samples than mock pool size."""
        from mobile_llm_benchmark.data_loaders import DataLoader
        loader = DataLoader(mock_mode=True)
        questions = loader.load("gsm8k", n_samples=100)
        assert len(questions) == 100

    def test_deterministic_seed(self):
        from mobile_llm_benchmark.data_loaders import DataLoader
        loader1 = DataLoader(mock_mode=True, seed=42)
        loader2 = DataLoader(mock_mode=True, seed=42)
        q1 = loader1.load("arc_challenge", n_samples=10)
        q2 = loader2.load("arc_challenge", n_samples=10)
        assert [q["question"] for q in q1] == [q["question"] for q in q2]


# =============================================================================
# Test 3: Scorers
# =============================================================================

class TestScorers:
    def test_gsm8k_correct(self):
        from mobile_llm_benchmark.scorers import GSM8KScorer
        scorer = GSM8KScorer()
        q = {"question": "2+2=?", "answer": "4"}
        assert scorer.score(q, "The answer is 4") is True
        assert scorer.score(q, "Answer: 4") is True
        assert scorer.score(q, "I think the answer is 5") is False

    def test_arc_correct(self):
        from mobile_llm_benchmark.scorers import ARCScorer
        scorer = ARCScorer()
        q = {"question": "...", "choices": ["A", "B", "C", "D"], "answer": "B"}
        assert scorer.score(q, "B") is True
        assert scorer.score(q, "The answer is B.") is True
        assert scorer.score(q, "C") is False

    def test_mmlu_correct(self):
        from mobile_llm_benchmark.scorers import MMLUScorer
        scorer = MMLUScorer()
        q = {"question": "?", "choices": ["a", "b", "c", "d"], "answer": "A", "subject": "test"}
        assert scorer.score(q, "A") is True
        assert scorer.score(q, "D") is False

    def test_hellaswag_correct(self):
        from mobile_llm_benchmark.scorers import HellaSwagScorer
        scorer = HellaSwagScorer()
        q = {"ctx": "...", "endings": ["a", "b", "c", "d"], "answer": "2"}
        assert scorer.score(q, "C") is True
        assert scorer.score(q, "A") is False

    def test_ifeval_scorer(self):
        from mobile_llm_benchmark.scorers import IFEvalScorer
        scorer = IFEvalScorer()
        q = {"prompt": "test", "check_fn": lambda r: "hello" in r.lower(), "instruction_id_list": [], "kwargs": []}
        assert scorer.score(q, "Hello world") is True
        assert scorer.score(q, "Goodbye world") is False

    def test_prompt_building(self):
        from mobile_llm_benchmark.scorers import get_scorer
        for bench_id in ["gsm8k", "arc_challenge", "mmlu", "hellaswag", "truthfulqa", "ifeval"]:
            scorer = get_scorer(bench_id)
            assert scorer is not None

    def test_unknown_benchmark_raises(self):
        from mobile_llm_benchmark.scorers import get_scorer
        with pytest.raises(ValueError):
            get_scorer("nonexistent_benchmark")


# =============================================================================
# Test 4: BenchmarkRunner (mock mode)
# =============================================================================

class TestBenchmarkRunner:
    def test_single_run_returns_result(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner, BenchmarkResult
        from mobile_llm_benchmark.config import MODELS
        runner = BenchmarkRunner(mock_mode=True)
        model = next(m for m in MODELS if m["name"] == "Llama-3.2-1B")
        result = runner.run_single(model, "arc_challenge", n_samples=20)
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "Llama-3.2-1B"
        assert result.benchmark == "arc_challenge"
        assert 0 <= result.accuracy <= 1
        assert result.n_samples == 20

    def test_result_has_token_fields(self):
        """BenchmarkResult now includes token usage fields."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS
        runner = BenchmarkRunner(mock_mode=True)
        result = runner.run_single(MODELS[0], "arc_challenge", n_samples=10)
        assert hasattr(result, "tokens_prompt")
        assert hasattr(result, "tokens_completion")
        assert hasattr(result, "tokens_total")
        assert hasattr(result, "avg_latency_ms")
        assert hasattr(result, "p95_latency_ms")
        # Mock mode populates simulated token counts
        assert result.tokens_prompt >= 0
        assert result.tokens_completion >= 0
        assert result.tokens_total == result.tokens_prompt + result.tokens_completion

    def test_result_latency_positive_in_mock(self):
        """Mock mode generates simulated positive latency values."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS
        runner = BenchmarkRunner(mock_mode=True)
        result = runner.run_single(MODELS[0], "gsm8k", n_samples=20)
        assert result.avg_latency_ms > 0
        assert result.p95_latency_ms >= result.avg_latency_ms

    def test_result_as_dict_has_token_keys(self):
        """as_dict() includes all new fields."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS
        runner = BenchmarkRunner(mock_mode=True)
        result = runner.run_single(MODELS[0], "arc_challenge", n_samples=10)
        d = result.as_dict()
        for key in ["tokens_prompt", "tokens_completion", "tokens_total", "avg_latency_ms", "p95_latency_ms"]:
            assert key in d, f"Missing key: {key}"

    def test_at_least_3_models_2_benchmarks(self):
        """At least 3 models × 2 benchmarks evaluated successfully."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS
        runner = BenchmarkRunner(mock_mode=True)
        models = MODELS[:3]
        bench_ids = [b["id"] for b in BENCHMARKS[:2]]
        results = runner.run(models, bench_ids, n_samples=10)
        assert len(results) >= 6, f"Expected ≥6 results, got {len(results)}"
        model_names = {r.model_name for r in results}
        assert len(model_names) >= 3
        benchmarks = {r.benchmark for r in results}
        assert len(benchmarks) >= 2

    def test_wilson_ci_ordering_all_results(self):
        """ci_lower < accuracy < ci_upper for all non-degenerate results."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS
        runner = BenchmarkRunner(mock_mode=True)
        models = MODELS[:3]
        bench_ids = [b["id"] for b in BENCHMARKS]
        results = runner.run(models, bench_ids, n_samples=30)
        for r in results:
            if 0 < r.accuracy < 1:
                assert r.ci_lower < r.accuracy, (
                    f"{r.model_name}/{r.benchmark}: ci_lower={r.ci_lower:.4f} >= accuracy={r.accuracy:.4f}"
                )
                assert r.accuracy < r.ci_upper, (
                    f"{r.model_name}/{r.benchmark}: accuracy={r.accuracy:.4f} >= ci_upper={r.ci_upper:.4f}"
                )

    def test_mock_flag_set(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS
        runner = BenchmarkRunner(mock_mode=True)
        result = runner.run_single(MODELS[0], "gsm8k", n_samples=10)
        assert result.mock is True

    def test_stream_yields_tuples(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS
        runner = BenchmarkRunner(mock_mode=True)
        models = MODELS[:2]
        bench_ids = [BENCHMARKS[0]["id"]]
        yielded = list(runner.run_stream(models, bench_ids, n_samples=5))
        assert len(yielded) > 0
        for item in yielded:
            assert len(item) == 2  # (results_list, log_line)

    def test_stream_log_has_timestamp(self):
        """Stream log lines include a timestamp for the new format."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS
        runner = BenchmarkRunner(mock_mode=True)
        models = MODELS[:1]
        bench_ids = [BENCHMARKS[0]["id"]]
        log_lines = [line for _, line in runner.run_stream(models, bench_ids, n_samples=5)]
        # At least one line should contain a time-like format HH:MM:SS
        import re
        time_pattern = re.compile(r"\d{2}:\d{2}:\d{2}")
        assert any(time_pattern.search(line) for line in log_lines), (
            f"No timestamp found in log lines: {log_lines}"
        )


# =============================================================================
# Test 4b: Config — scenario presets (new)
# =============================================================================

class TestScenarioPresets:
    def test_presets_defined(self):
        """SCENARIO_PRESETS is non-empty and has expected keys."""
        from mobile_llm_benchmark.config import SCENARIO_PRESETS
        assert len(SCENARIO_PRESETS) >= 3
        for key, preset in SCENARIO_PRESETS.items():
            assert "name" in preset
            assert "benchmarks" in preset
            assert "n_samples" in preset
            assert "models" in preset
            assert isinstance(preset["benchmarks"], list)
            assert len(preset["benchmarks"]) >= 1

    def test_preset_benchmarks_valid(self):
        """All preset benchmark IDs are valid."""
        from mobile_llm_benchmark.config import SCENARIO_PRESETS, BENCHMARK_BY_ID
        for key, preset in SCENARIO_PRESETS.items():
            for bid in preset["benchmarks"]:
                assert bid in BENCHMARK_BY_ID, (
                    f"Preset '{key}' references unknown benchmark '{bid}'"
                )

    def test_preset_models_valid(self):
        """All preset model names exist in MODELS."""
        from mobile_llm_benchmark.config import SCENARIO_PRESETS, MODEL_BY_NAME
        for key, preset in SCENARIO_PRESETS.items():
            for mname in preset["models"]:
                assert mname in MODEL_BY_NAME, (
                    f"Preset '{key}' references unknown model '{mname}'"
                )

    def test_qwen3_06b_in_models(self):
        """Qwen3-0.6B is in the model registry."""
        from mobile_llm_benchmark.config import MODEL_BY_NAME
        assert "Qwen3-0.6B" in MODEL_BY_NAME
        assert MODEL_BY_NAME["Qwen3-0.6B"]["params"] == "0.6B"

    def test_model_capabilities_have_qwen3_06b(self):
        from mobile_llm_benchmark.config import MODEL_CAPABILITIES
        assert "Qwen3-0.6B" in MODEL_CAPABILITIES


# =============================================================================
# Test 5: CSV output has required columns
# =============================================================================

class TestCSVOutput:
    def test_csv_columns(self):
        """outputs/benchmark_results.csv must have required columns."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS[:3]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            df = rg.results_to_df(results)
            csv_path = rg.save_csv(df)

            assert csv_path.exists(), f"CSV not created at {csv_path}"
            loaded = pd.read_csv(csv_path)
            required_cols = {"model", "benchmark", "accuracy", "ci_lower", "ci_upper", "n_samples"}
            missing = required_cols - set(loaded.columns)
            assert not missing, f"Missing columns: {missing}"

    def test_csv_has_token_columns(self):
        """CSV includes new token usage and latency columns."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            df = rg.results_to_df(results)
            csv_path = rg.save_csv(df)
            loaded = pd.read_csv(csv_path)
            for col in ["tokens_total", "avg_latency_ms"]:
                assert col in loaded.columns, f"Missing new column: {col}"

    def test_csv_ci_ordering(self):
        """All rows in CSV: ci_lower < accuracy < ci_upper (non-degenerate)."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS], n_samples=30)
            rg = ReportGenerator(output_dir=tmpdir)
            df = rg.results_to_df(results)
            csv_path = rg.save_csv(df)
            loaded = pd.read_csv(csv_path)

            for _, row in loaded.iterrows():
                acc = row["accuracy"]
                if 0 < acc < 1:
                    assert row["ci_lower"] < acc, (
                        f"ci_lower {row['ci_lower']:.4f} >= accuracy {acc:.4f}"
                    )
                    assert acc < row["ci_upper"], (
                        f"accuracy {acc:.4f} >= ci_upper {row['ci_upper']:.4f}"
                    )


# =============================================================================
# Test 5b: JSON export (new)
# =============================================================================

class TestJSONExport:
    def test_json_file_created(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            json_path = rg.save_json(results)
            assert json_path.exists()
            assert json_path.suffix == ".json"

    def test_json_structure(self):
        """JSON has expected top-level keys and valid content."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            json_path = rg.save_json(results)

            data = json.loads(json_path.read_text())
            for key in ["generated_at", "n_models", "n_benchmarks", "mock", "total_tokens", "models", "raw_results"]:
                assert key in data, f"Missing JSON key: {key}"

            assert data["n_models"] == 2
            assert data["n_benchmarks"] == 2
            assert isinstance(data["models"], dict)
            assert len(data["raw_results"]) == 4  # 2 models × 2 benchmarks

    def test_json_model_has_avg_accuracy(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            json_path = rg.save_json(results)

            data = json.loads(json_path.read_text())
            for model_name, summary in data["models"].items():
                assert "avg_accuracy" in summary
                assert 0 <= summary["avg_accuracy"] <= 1

    def test_json_is_valid(self):
        """JSON output can be parsed without errors."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS[:3]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)
            json_path = paths.get("json")
            assert json_path is not None and json_path.exists()
            parsed = json.loads(json_path.read_text())
            assert isinstance(parsed, dict)


# =============================================================================
# Test 6: HTML leaderboard
# =============================================================================

class TestHTMLLeaderboard:
    def test_leaderboard_html_exists(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS[:3]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)
            lb_path = paths["leaderboard_html"]
            assert lb_path.exists()

    def test_leaderboard_html_is_valid(self):
        """HTML contains a table and model names."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            models = MODELS[:3]
            results = runner.run(models, [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)
            html = paths["leaderboard_html"].read_text(encoding="utf-8")

            assert "<table" in html.lower()
            assert "</table>" in html.lower()
            assert any(m["name"] in html for m in models)

    def test_report_html_exists(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)
            assert paths["report_html"].exists()
            html = paths["report_html"].read_text()
            assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()

    def test_report_html_has_stats_grid(self):
        """New HTML report contains the stats summary grid."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)
            html = paths["report_html"].read_text()
            assert "stats-grid" in html
            assert "stat-card" in html


# =============================================================================
# Test 7: Charts
# =============================================================================

class TestCharts:
    def test_radar_chart_exists(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            radar_path = rg.save_radar_chart(results)
            assert radar_path.exists()
            assert radar_path.suffix == ".png"
            assert radar_path.stat().st_size > 1000

    def test_radar_chart_is_valid_png(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            radar_path = rg.save_radar_chart(results)
            header = radar_path.read_bytes()[:8]
            assert header[:4] == b"\x89PNG"

    def test_timing_chart_created(self):
        """Timing chart is generated when latency data exists (mock mode)."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            timing_path = rg.save_timing_chart(results)
            # Mock results have simulated latencies, so chart should be created
            assert timing_path is not None
            assert timing_path.exists()
            assert timing_path.suffix == ".png"

    def test_timing_chart_valid_png(self):
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:2], [b["id"] for b in BENCHMARKS[:2]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            timing_path = rg.save_timing_chart(results)
            if timing_path and timing_path.exists():
                assert timing_path.read_bytes()[:4] == b"\x89PNG"


# =============================================================================
# Test 8: Gradio UI structure
# =============================================================================

class TestGradioUI:
    def test_app_imports(self):
        """app.py can be imported without error."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app",
            Path(__file__).parent.parent / "app.py",
        )
        assert spec is not None

    def test_build_ui_returns_blocks(self):
        """build_ui() returns a gr.Blocks instance with expected components."""
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        assert isinstance(ui, gr.Blocks)

    @staticmethod
    def _get_all_blocks(ui):
        import gradio as gr
        blocks = {}
        if hasattr(ui, "blocks"):
            blocks = ui.blocks
        elif hasattr(ui, "fns"):
            blocks = {}
        return blocks

    @staticmethod
    def _normalize_choices(choices):
        result = []
        for c in choices or []:
            if isinstance(c, (list, tuple)):
                result.append(str(c[0]))
            else:
                result.append(str(c))
        return result

    def test_ui_has_model_checkboxes(self):
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        cg_labels = []
        for block in self._get_all_blocks(ui).values():
            if isinstance(block, gr.CheckboxGroup):
                cg_labels.append(getattr(block, "label", ""))
        assert any("model" in (label or "").lower() for label in cg_labels), (
            f"No model CheckboxGroup found. Labels: {cg_labels}"
        )

    def test_ui_has_benchmark_checkboxes(self):
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        cg_labels = []
        for block in self._get_all_blocks(ui).values():
            if isinstance(block, gr.CheckboxGroup):
                cg_labels.append(getattr(block, "label", ""))
        assert any("benchmark" in (label or "").lower() for label in cg_labels), (
            f"No benchmark CheckboxGroup found. Labels: {cg_labels}"
        )

    def test_ui_has_sample_count_slider(self):
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        sliders = [
            block for block in self._get_all_blocks(ui).values()
            if isinstance(block, gr.Slider)
        ]
        assert len(sliders) >= 1, "No Slider component found"
        slider = sliders[0]
        min_val = getattr(slider, "minimum", None) or getattr(slider, "min", None) or 0
        max_val = getattr(slider, "maximum", None) or getattr(slider, "max", None) or 0
        assert min_val >= 1
        assert max_val >= 50

    def test_ui_has_run_button(self):
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        buttons = [
            block for block in self._get_all_blocks(ui).values()
            if isinstance(block, gr.Button)
        ]
        assert len(buttons) >= 1, "No Button component found"
        btn_values = [getattr(b, "value", "") or "" for b in buttons]
        assert any(
            "run" in v.lower() or "benchmark" in v.lower()
            for v in btn_values
        ), f"No Run button found. Button values: {btn_values}"

    def test_ui_has_compare_button(self):
        """UI has a Compare button for the new head-to-head tab."""
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        buttons = [
            block for block in self._get_all_blocks(ui).values()
            if isinstance(block, gr.Button)
        ]
        btn_values = [getattr(b, "value", "") or "" for b in buttons]
        assert any(
            "compare" in v.lower()
            for v in btn_values
        ), f"No Compare button found. Button values: {btn_values}"

    def test_ui_has_dropdowns_for_comparison(self):
        """UI includes Dropdowns for model A/B selection in comparison tab."""
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        dropdowns = [
            block for block in self._get_all_blocks(ui).values()
            if isinstance(block, gr.Dropdown)
        ]
        assert len(dropdowns) >= 2, f"Expected ≥2 Dropdown components, got {len(dropdowns)}"

    def test_model_choices_complete(self):
        """Model CheckboxGroup contains all 10 configured models."""
        import gradio as gr
        import app as app_module
        from mobile_llm_benchmark.config import MODELS
        ui = app_module.build_ui()
        for block in self._get_all_blocks(ui).values():
            if isinstance(block, gr.CheckboxGroup):
                label = getattr(block, "label", "") or ""
                if "model" in label.lower():
                    choices = self._normalize_choices(block.choices)
                    model_names = [m["name"] for m in MODELS]
                    for name in model_names:
                        assert name in choices, f"Model {name!r} missing from selector"
                    break

    def test_benchmark_choices_complete(self):
        """Benchmark CheckboxGroup contains all 6 benchmarks."""
        import gradio as gr
        import app as app_module
        from mobile_llm_benchmark.config import BENCHMARK_NAMES
        ui = app_module.build_ui()
        for block in self._get_all_blocks(ui).values():
            if isinstance(block, gr.CheckboxGroup):
                label = getattr(block, "label", "") or ""
                if "benchmark" in label.lower():
                    choices = self._normalize_choices(block.choices)
                    for name in BENCHMARK_NAMES:
                        assert name in choices, f"Benchmark {name!r} missing from selector"
                    break

    def test_ui_has_json_download(self):
        """Download tab includes a JSON file output component."""
        import gradio as gr
        import app as app_module
        ui = app_module.build_ui()
        file_components = [
            block for block in self._get_all_blocks(ui).values()
            if isinstance(block, gr.File)
        ]
        file_labels = [getattr(f, "label", "") or "" for f in file_components]
        assert any("json" in label.lower() for label in file_labels), (
            f"No JSON File component found. Labels: {file_labels}"
        )


# =============================================================================
# Test 9: Model client helpers (new)
# =============================================================================

class TestModelClientHelpers:
    def test_generate_result_dataclass(self):
        from mobile_llm_benchmark.model_client import GenerateResult
        r = GenerateResult(
            text="hello",
            tokens_prompt=10,
            tokens_completion=5,
            latency_ms=250.0,
            model_id="test/model",
            provider="openrouter",
        )
        assert r.tokens_total == 15
        assert r.text == "hello"
        assert r.latency_ms == 250.0

    def test_is_rate_limit_error(self):
        from mobile_llm_benchmark.model_client import _is_rate_limit_error
        assert _is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))
        assert _is_rate_limit_error(Exception("rate limit exceeded"))
        assert not _is_rate_limit_error(Exception("HTTP 500 Internal Server Error"))
        assert not _is_rate_limit_error(Exception("timeout"))

    def test_backoff_wait_increases_with_attempt(self):
        from mobile_llm_benchmark.model_client import _backoff_wait
        # With fixed seed-like behavior, later attempts should be longer on average
        waits = [_backoff_wait(i) for i in range(4)]
        # Each successive wait should be at least as long as 1/2 the previous
        assert waits[1] > waits[0] * 0.5
        assert waits[2] > waits[1] * 0.5

    def test_backoff_wait_rate_limited_is_longer(self):
        from mobile_llm_benchmark.model_client import _backoff_wait
        normal = _backoff_wait(0, rate_limited=False)
        rate_limited = _backoff_wait(0, rate_limited=True)
        assert rate_limited >= normal


# =============================================================================
# Test 10: demo.py integration (mock mode)
# =============================================================================

class TestDemoIntegration:
    def test_demo_produces_csv(self):
        """Running demo logic (mock) produces CSV with correct columns."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            demo_models = [m for m in MODELS if m["name"] in ("Llama-3.2-1B", "SmolLM3-3B", "Qwen2.5-3B")]
            bench_ids = [b["id"] for b in BENCHMARKS]

            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(demo_models, bench_ids, n_samples=20)

            assert len(results) >= 3 * 6, f"Expected ≥18 results, got {len(results)}"

            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)

            csv = pd.read_csv(paths["csv"])
            required = {"model", "benchmark", "accuracy", "ci_lower", "ci_upper", "n_samples"}
            assert required <= set(csv.columns)

            for _, row in csv.iterrows():
                acc = row["accuracy"]
                if 0 < acc < 1:
                    assert row["ci_lower"] < acc
                    assert acc < row["ci_upper"]

    def test_demo_produces_all_artifacts(self):
        """All required output files are created."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.report_generator import ReportGenerator
        from mobile_llm_benchmark.config import MODELS, BENCHMARKS

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(mock_mode=True)
            results = runner.run(MODELS[:3], [b["id"] for b in BENCHMARKS[:3]], n_samples=10)
            rg = ReportGenerator(output_dir=tmpdir)
            paths = rg.generate_all(results)

            assert paths["csv"].exists()
            assert paths["leaderboard_html"].exists()
            assert paths["report_html"].exists()
            assert paths["radar_chart"].exists()
            assert paths["bar_chart"].exists()
            assert paths["json"].exists()

    def test_demo_scenario_quick_runs(self):
        """The 'quick' scenario preset runs end-to-end in mock mode."""
        from mobile_llm_benchmark.runner import BenchmarkRunner
        from mobile_llm_benchmark.config import MODELS, SCENARIO_PRESETS, BENCHMARK_BY_ID

        preset = SCENARIO_PRESETS["quick"]
        demo_models = [m for m in MODELS if m["name"] in preset["models"]]
        bench_ids = preset["benchmarks"]

        runner = BenchmarkRunner(mock_mode=True)
        results = runner.run(demo_models, bench_ids, n_samples=preset["n_samples"])
        assert len(results) >= len(demo_models) * len(bench_ids)

    def test_demo_dry_run_exits_cleanly(self):
        """demo.py --dry-run exits without calling any APIs."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "demo.py", "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"dry-run failed:\n{result.stderr}"
        assert "dry-run" in result.stdout.lower() or "no calls" in result.stdout.lower()
