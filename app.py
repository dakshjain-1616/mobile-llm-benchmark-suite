#!/usr/bin/env python3
"""Gradio web UI for the Mobile LLM Benchmark Suite.

Usage:
    python app.py
    # or with custom host/port:
    GRADIO_HOST=127.0.0.1 GRADIO_PORT=8080 python app.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Imports ────────────────────────────────────────────────────────────────────
import gradio as gr
import pandas as pd

from mobile_llm_benchmark.config import (
    MODELS, BENCHMARKS, BENCHMARK_NAMES, BENCHMARK_IDS, SCENARIO_PRESETS
)
from mobile_llm_benchmark.runner import BenchmarkRunner, BenchmarkResult
from mobile_llm_benchmark.report_generator import ReportGenerator
from mobile_llm_benchmark.statistical import pairwise_significance

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

MODEL_NAMES = [m["name"] for m in MODELS]
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _is_mock_mode(provider: str) -> bool:
    if provider == "mock":
        return True
    if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        return True
    if provider == "hf" and not os.getenv("HUGGINGFACE_TOKEN"):
        return True
    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")):
        return True
    return False


def _results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["Model", "Benchmark", "Accuracy %", "CI Lower %", "CI Upper %", "N", "Tokens", "Avg Latency (ms)"])
    rows = []
    for r in results:
        rows.append({
            "Model": r.model_name,
            "Benchmark": r.benchmark_name,
            "Accuracy %": round(r.accuracy * 100, 1),
            "CI Lower %": round(r.ci_lower * 100, 1),
            "CI Upper %": round(r.ci_upper * 100, 1),
            "N": r.n_samples,
            "Tokens": r.tokens_total if r.tokens_total > 0 else "—",
            "Avg Latency (ms)": round(r.avg_latency_ms, 0) if r.avg_latency_ms > 0 else "—",
        })
    return pd.DataFrame(rows)


def _build_leaderboard_df(results: list[BenchmarkResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    from collections import defaultdict
    import numpy as np

    model_scores: dict[str, list[float]] = defaultdict(list)
    model_tokens: dict[str, int] = defaultdict(int)
    for r in results:
        model_scores[r.model_name].append(r.accuracy)
        model_tokens[r.model_name] += r.tokens_total

    rows = []
    for model, accs in model_scores.items():
        tok = model_tokens[model]
        rows.append({
            "Model": model,
            "Avg Accuracy %": round(float(np.mean(accs)) * 100, 1),
            "N Benchmarks": len(accs),
            "Total Tokens": f"{tok:,}" if tok > 0 else "—",
        })
    df = pd.DataFrame(rows).sort_values("Avg Accuracy %", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def _build_token_stats(results: list[BenchmarkResult]) -> str:
    """Return a formatted live stats panel string showing token usage, latency, and evaluation count."""
    if not results:
        return ""
    total_prompt = sum(r.tokens_prompt for r in results)
    total_completion = sum(r.tokens_completion for r in results)
    total = total_prompt + total_completion
    n_evals = len(results)

    lat_values = [r.avg_latency_ms for r in results if r.avg_latency_ms > 0]
    lat_str = f"{sum(lat_values)/len(lat_values):.0f} ms" if lat_values else "—"

    tok_str = f"{total:,} ({total_prompt:,} prompt + {total_completion:,} completion)" if total > 0 else "—"

    return (
        f"| **Evaluations** | **Tokens used** | **Avg latency** |\n"
        f"|---|---|---|\n"
        f"| {n_evals} turns | {tok_str} | {lat_str} |"
    )


# ---------------------------------------------------------------------------
# Core benchmark function (generator for streaming)
# ---------------------------------------------------------------------------

def run_benchmarks_stream(
    selected_models: list[str],
    selected_benchmarks: list[str],
    n_samples: int,
    provider: str,
    progress: gr.Progress = gr.Progress(),
):
    """Generator yielding (log, results_df, leaderboard_df, radar_img, bar_img, timing_img, report_file, csv_file, json_file, token_stats)."""

    if not selected_models:
        yield "⚠ Please select at least one model.", None, None, None, None, None, None, None, None, ""
        return
    if not selected_benchmarks:
        yield "⚠ Please select at least one benchmark.", None, None, None, None, None, None, None, None, ""
        return

    model_configs = [m for m in MODELS if m["name"] in selected_models]
    name_to_id = {b["name"]: b["id"] for b in BENCHMARKS}
    bench_ids = [name_to_id.get(b, b) for b in selected_benchmarks]

    mock_mode = _is_mock_mode(provider)
    mode_str = "MOCK" if mock_mode else f"LIVE ({provider.upper()})"

    log = (
        f"🚀 Starting benchmark — {len(model_configs)} models × {len(bench_ids)} benchmarks × "
        f"{n_samples} samples [{mode_str}]\n"
    )
    yield log, None, None, None, None, None, None, None, None, ""

    runner = BenchmarkRunner(mock_mode=mock_mode)
    all_results: list[BenchmarkResult] = []
    total = len(model_configs) * len(bench_ids)

    for results_so_far, line in runner.run_stream(model_configs, bench_ids, n_samples):
        all_results = results_so_far
        log += line + "\n"
        pct = len(all_results) / total if total > 0 else 0
        progress(pct, desc=line[:80])

        token_stats = _build_token_stats(all_results)
        if all_results:
            results_df = _results_to_dataframe(all_results)
            lb_df = _build_leaderboard_df(all_results)
        else:
            results_df, lb_df = None, None

        yield log, results_df, lb_df, None, None, None, None, None, None, token_stats

    if not all_results:
        yield log + "\n❌ No results produced.", None, None, None, None, None, None, None, None, ""
        return

    log += "\n📊 Generating report artifacts...\n"
    yield log, _results_to_dataframe(all_results), _build_leaderboard_df(all_results), None, None, None, None, None, None, _build_token_stats(all_results)

    try:
        rg = ReportGenerator(output_dir=OUTPUT_DIR)
        paths = rg.generate_all(all_results)
        log += "✅ Reports generated.\n"

        radar_img = str(paths.get("radar_chart", "")) or None
        bar_img = str(paths.get("bar_chart", "")) or None
        timing_img = str(paths.get("timing_chart", "")) or None
        html_file = str(paths.get("report_html", "")) or None
        csv_file = str(paths.get("csv", "")) or None
        json_file = str(paths.get("json", "")) or None

        log += "\n📁 Outputs:\n"
        for name, path in paths.items():
            log += f"  {name}: {path}\n"

    except Exception as exc:
        log += f"⚠ Report generation failed: {exc}\n"
        radar_img = bar_img = timing_img = html_file = csv_file = json_file = None

    token_stats = _build_token_stats(all_results)
    progress(1.0, desc="Complete!")
    yield (
        log,
        _results_to_dataframe(all_results),
        _build_leaderboard_df(all_results),
        radar_img,
        bar_img,
        timing_img,
        html_file,
        csv_file,
        json_file,
        token_stats,
    )


# ---------------------------------------------------------------------------
# Model comparison function
# ---------------------------------------------------------------------------

def compare_models(
    model_a: str,
    model_b: str,
    selected_benchmarks: list[str],
    n_samples: int,
    provider: str,
    progress: gr.Progress = gr.Progress(),
):
    """Head-to-head model comparison with significance testing."""
    if not model_a or not model_b:
        yield "⚠ Please select two models to compare.", None
        return
    if model_a == model_b:
        yield "⚠ Please select two different models.", None
        return
    if not selected_benchmarks:
        yield "⚠ Please select at least one benchmark.", None
        return

    model_configs = [m for m in MODELS if m["name"] in (model_a, model_b)]
    name_to_id = {b["name"]: b["id"] for b in BENCHMARKS}
    bench_ids = [name_to_id.get(b, b) for b in selected_benchmarks]

    mock_mode = _is_mock_mode(provider)
    log = f"⚔️  Comparing {model_a} vs {model_b} [{len(bench_ids)} benchmarks, {'mock' if mock_mode else 'live'}]\n"
    yield log, None

    runner = BenchmarkRunner(mock_mode=mock_mode)
    all_results: list[BenchmarkResult] = []
    total = len(model_configs) * len(bench_ids)

    for results_so_far, line in runner.run_stream(model_configs, bench_ids, n_samples):
        all_results = results_so_far
        log += line + "\n"
        progress(len(all_results) / total if total > 0 else 0, desc=line[:80])
        yield log, None

    if not all_results:
        yield log + "\n❌ No results produced.", None
        return

    # Build comparison table with significance tests
    rows = []
    for bench_id in bench_ids:
        bench_results = [r for r in all_results if r.benchmark == bench_id]
        r_a = next((r for r in bench_results if r.model_name == model_a), None)
        r_b = next((r for r in bench_results if r.model_name == model_b), None)
        if r_a and r_b:
            sig_tests = pairwise_significance(bench_results, bench_id)
            pair = next(
                (s for s in sig_tests if {s["model_a"], s["model_b"]} == {model_a, model_b}),
                None,
            )
            winner = "=" if not pair or not pair["significant"] else (
                model_a if r_a.accuracy > r_b.accuracy else model_b
            )
            rows.append({
                "Benchmark": r_a.benchmark_name,
                f"{model_a} Acc%": round(r_a.accuracy * 100, 1),
                f"{model_b} Acc%": round(r_b.accuracy * 100, 1),
                "Δ (pp)": round((r_a.accuracy - r_b.accuracy) * 100, 1),
                "Cohen's h": round(pair["cohens_h"], 3) if pair else "—",
                "Effect": pair["effect_size"] if pair else "—",
                "p-value": round(pair["p_value"], 3) if pair else "—",
                "Winner": winner,
            })

    df = pd.DataFrame(rows) if rows else None
    log += "\n✅ Comparison complete.\n"
    progress(1.0, desc="Done!")
    yield log, df


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    mock_note = ""
    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")):
        mock_note = " ⚠ No API keys detected — will use mock data. Set OPENROUTER_API_KEY or HUGGINGFACE_TOKEN for real results."

    with gr.Blocks(
        title="Mobile LLM Benchmark Suite",
        theme=gr.themes.Soft(primary_hue="purple"),
        css="""
            .title-header { text-align: center; margin-bottom: 1rem; }
            .run-btn { background: linear-gradient(135deg, #7c3aed, #2563eb) !important; }
            .stats-panel { background: #f8f7ff; border: 1px solid #e2e0ff;
                           border-radius: 8px; padding: 0.6rem 1rem; margin-top: 0.5rem; }
            .neo-header { text-align: center; padding: 0.5rem 0 1rem 0; }
        """,
    ) as demo:
        gr.Markdown(
            "# 📱 Mobile LLM Benchmark Suite\n"
            "> Benchmark 1–4B parameter LLMs across 6 tasks with **Wilson 95% CI** and **Cohen's h** effect sizes — no GPU needed.\n\n"
            "<div style='text-align:center; font-size:0.85rem; color:#6b7280'>"
            "Built autonomously by <a href='https://heyneo.so' target='_blank'><strong>NEO</strong></a> — your autonomous AI Agent"
            "</div>"
            + (f"\n\n> ⚠️ {mock_note.strip()}" if mock_note else ""),
            elem_classes=["neo-header"],
        )

        with gr.Tabs():
            # ----------------------------------------------------------
            # Tab 1: Configure & Run
            # ----------------------------------------------------------
            with gr.Tab("⚙️ Configure & Run"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_selector = gr.CheckboxGroup(
                            choices=MODEL_NAMES,
                            value=["Llama-3.2-1B", "SmolLM2-1.7B", "Qwen2.5-3B"],
                            label="Models",
                            info="Select models to benchmark",
                        )
                        benchmark_selector = gr.CheckboxGroup(
                            choices=BENCHMARK_NAMES,
                            value=BENCHMARK_NAMES,
                            label="Benchmarks",
                            info="Select benchmarks to run",
                        )
                        n_samples_slider = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=50,
                            step=10,
                            label="Samples per benchmark",
                            info="More samples = tighter confidence intervals (slower)",
                        )
                        provider_radio = gr.Radio(
                            choices=[
                                ("🔀 Auto — use whichever API key is set", "auto"),
                                ("🌐 OpenRouter — requires OPENROUTER_API_KEY", "openrouter"),
                                ("🤗 HuggingFace — requires HUGGINGFACE_TOKEN", "hf"),
                                ("🎭 Mock — synthetic data, always works, no key needed", "mock"),
                            ],
                            value="auto",
                            label="API Provider",
                        )
                        run_btn = gr.Button(
                            "🚀 Run Benchmark",
                            variant="primary",
                            elem_classes=["run-btn"],
                        )

                    with gr.Column(scale=2):
                        log_output = gr.Textbox(
                            label="Live Progress Log",
                            lines=18,
                            max_lines=40,
                            interactive=False,
                            placeholder="Click 'Run Benchmark' to start...",
                        )
                        with gr.Group(elem_classes=["stats-panel"]):
                            gr.Markdown("**📊 Live Stats**")
                            token_stats_display = gr.Markdown(
                                value="_Stats appear here during the run._",
                            )

            # ----------------------------------------------------------
            # Tab 2: Scenario Cards
            # ----------------------------------------------------------
            with gr.Tab("🎯 Scenarios"):
                gr.Markdown(
                    "### Quick-start scenario presets\n"
                    "Click a scenario to pre-fill the Configure & Run settings, then press **Run Benchmark**."
                )

                scenario_status = gr.Markdown(value="")

                with gr.Row():
                    for preset_id, preset in SCENARIO_PRESETS.items():
                        with gr.Column():
                            gr.Markdown(
                                f"**{preset['emoji']} {preset['name']}**\n\n"
                                f"{preset['description']}\n\n"
                                f"- Benchmarks: {', '.join(preset['benchmarks'])}\n"
                                f"- Samples: {preset['n_samples']}\n"
                                f"- Models: {len(preset['models'])} selected"
                            )
                            gr.Button(
                                f"Load {preset['name']}",
                                variant="secondary",
                                elem_id=f"scenario_{preset_id}",
                            ).click(
                                fn=_make_scenario_loader(preset),
                                outputs=[
                                    model_selector,
                                    benchmark_selector,
                                    n_samples_slider,
                                    scenario_status,
                                ],
                            )

            # ----------------------------------------------------------
            # Tab 3: Results
            # ----------------------------------------------------------
            with gr.Tab("📊 Results"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🏆 Leaderboard (by avg accuracy)")
                        leaderboard_output = gr.DataFrame(
                            label="Leaderboard",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 All Results")
                        results_df_output = gr.DataFrame(
                            label="Detailed Results",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        radar_output = gr.Image(
                            label="Radar Chart — Per-Model Profile",
                            type="filepath",
                        )
                    with gr.Column():
                        bar_output = gr.Image(
                            label="Per-Benchmark Accuracy (with 95% CI)",
                            type="filepath",
                        )
                with gr.Row():
                    with gr.Column():
                        timing_output = gr.Image(
                            label="Model Response Latency",
                            type="filepath",
                        )

            # ----------------------------------------------------------
            # Tab 4: Compare Models
            # ----------------------------------------------------------
            with gr.Tab("⚔️ Compare Models"):
                gr.Markdown(
                    "### Head-to-head model comparison\n"
                    "Select two models and run a statistical significance test across benchmarks."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_model_a = gr.Dropdown(
                            choices=MODEL_NAMES,
                            value=MODEL_NAMES[0] if MODEL_NAMES else None,
                            label="Model A",
                        )
                        compare_model_b = gr.Dropdown(
                            choices=MODEL_NAMES,
                            value=MODEL_NAMES[1] if len(MODEL_NAMES) > 1 else None,
                            label="Model B",
                        )
                        compare_benchmarks = gr.CheckboxGroup(
                            choices=BENCHMARK_NAMES,
                            value=BENCHMARK_NAMES,
                            label="Benchmarks",
                        )
                        compare_n_samples = gr.Slider(
                            minimum=10, maximum=200, value=50, step=10,
                            label="Samples per benchmark",
                        )
                        compare_provider = gr.Radio(
                            choices=[
                                ("🔀 Auto", "auto"),
                                ("🌐 OpenRouter", "openrouter"),
                                ("🤗 HuggingFace", "hf"),
                                ("🎭 Mock", "mock"),
                            ],
                            value="auto",
                            label="API Provider",
                        )
                        compare_btn = gr.Button("⚔️ Compare", variant="primary")

                    with gr.Column(scale=2):
                        compare_log = gr.Textbox(
                            label="Comparison Log",
                            lines=12,
                            max_lines=20,
                            interactive=False,
                        )
                        compare_table = gr.DataFrame(
                            label="Head-to-Head Results (with significance tests)",
                            interactive=False,
                        )

            # ----------------------------------------------------------
            # Tab 5: Download
            # ----------------------------------------------------------
            with gr.Tab("⬇️ Download"):
                gr.Markdown(
                    "### Download report files\n"
                    "*(Available after benchmark completes in the Configure & Run tab)*"
                )
                with gr.Row():
                    html_report_file = gr.File(label="HTML Report", file_types=[".html"])
                    csv_results_file = gr.File(label="CSV Results", file_types=[".csv"])
                    json_results_file = gr.File(label="JSON Results", file_types=[".json"])

        # ------------------------------------------------------------------
        # Wire up the Run button
        # ------------------------------------------------------------------
        run_btn.click(
            fn=run_benchmarks_stream,
            inputs=[model_selector, benchmark_selector, n_samples_slider, provider_radio],
            outputs=[
                log_output,
                results_df_output,
                leaderboard_output,
                radar_output,
                bar_output,
                timing_output,
                html_report_file,
                csv_results_file,
                json_results_file,
                token_stats_display,
            ],
            show_progress="hidden",
        )

        # Wire up Compare button
        compare_btn.click(
            fn=compare_models,
            inputs=[compare_model_a, compare_model_b, compare_benchmarks, compare_n_samples, compare_provider],
            outputs=[compare_log, compare_table],
            show_progress="hidden",
        )

    return demo


def _make_scenario_loader(preset: dict):
    """Return a closure that loads a scenario preset into the UI controls."""
    def _load():
        bench_name_map = {b["id"]: b["name"] for b in BENCHMARKS}
        bench_names = [bench_name_map.get(b, b) for b in preset["benchmarks"]]
        return (
            preset["models"],
            bench_names,
            preset["n_samples"],
            f"✅ Loaded scenario: **{preset['emoji']} {preset['name']}** — {preset['description']}",
        )
    return _load


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("GRADIO_HOST", "0.0.0.0")
    port = int(os.getenv("GRADIO_PORT", "7860"))

    logger.info("Starting Gradio UI at http://%s:%d", host, port)
    ui = build_ui()
    ui.launch(
        server_name=host,
        server_port=port,
        show_error=True,
    )
