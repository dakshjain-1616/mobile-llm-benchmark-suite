#!/usr/bin/env python3
"""Demo script — runs configurable model × benchmark combinations.

Auto-detects mock mode when no API keys are set.
Saves outputs to the OUTPUT_DIR directory.

Usage examples:
    python demo.py                                         # mock mode, default 3 models
    python demo.py --scenario quick                        # preset: 2 benchmarks, 20 samples
    python demo.py --models Phi-4-Mini,Gemma-3-4B          # specific models
    python demo.py --benchmarks gsm8k,arc_challenge        # specific benchmarks
    python demo.py --n-samples 100                         # more samples → tighter CIs
    python demo.py --provider openrouter                   # force OpenRouter API
    python demo.py --dry-run                               # validate config, no API calls
    OPENROUTER_API_KEY=... python demo.py --scenario math  # real API, math scenario
"""

from __future__ import annotations

import argparse
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

# ── Logging ───────────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _detect_mock_mode(provider: str) -> bool:
    if provider == "mock":
        return True
    forced = os.getenv("MOCK_MODE", "false").lower()
    if forced == "true":
        return True
    if forced == "false" and (os.getenv("OPENROUTER_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")):
        return False
    return True


def _parse_args() -> argparse.Namespace:
    from mobile_llm_benchmark.config import MODELS, SCENARIO_PRESETS

    model_names = [m["name"] for m in MODELS]
    scenario_keys = list(SCENARIO_PRESETS.keys())

    parser = argparse.ArgumentParser(
        description="Mobile LLM Benchmark Suite — CLI demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Scenario presets: {', '.join(scenario_keys)}
Available models: {', '.join(model_names)}
        """,
    )
    parser.add_argument(
        "--scenario",
        choices=scenario_keys,
        default=None,
        metavar="PRESET",
        help=f"Use a preset scenario ({', '.join(scenario_keys)}). Overrides --models/--benchmarks/--n-samples.",
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="M1,M2,...",
        help="Comma-separated model names (default: Llama-3.2-1B,SmolLM3-3B,Qwen2.5-3B)",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        metavar="B1,B2,...",
        help="Comma-separated benchmark IDs (default: all 6)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help=f"Samples per benchmark (default: {os.getenv('DEMO_N_SAMPLES', '50')})",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openrouter", "hf", "mock"],
        default="auto",
        help="API provider (default: auto — picks whichever key is set)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "outputs"),
        metavar="DIR",
        help="Output directory for artifacts (default: outputs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print what would run, without making API calls",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation (faster for quick checks)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        rich_available = True
    except ImportError:
        rich_available = False

    from mobile_llm_benchmark.config import MODELS, BENCHMARKS, SCENARIO_PRESETS

    # ── Resolve settings: scenario overrides individual flags ────────────────
    if args.scenario:
        preset = SCENARIO_PRESETS[args.scenario]
        demo_model_names = preset["models"]
        demo_bench_ids = preset["benchmarks"]
        n_samples = args.n_samples or preset["n_samples"]
        scenario_note = f" [Scenario: {preset['emoji']} {preset['name']}]"
    else:
        demo_model_names = (
            [m.strip() for m in args.models.split(",")]
            if args.models
            else ["Llama-3.2-1B", "SmolLM3-3B", "Qwen2.5-3B"]
        )
        demo_bench_ids = (
            [b.strip() for b in args.benchmarks.split(",")]
            if args.benchmarks
            else [b["id"] for b in BENCHMARKS]
        )
        n_samples = args.n_samples or int(os.getenv("DEMO_N_SAMPLES", "50"))
        scenario_note = ""

    mock_mode = _detect_mock_mode(args.provider)
    output_dir = args.output_dir

    # ── Validate model names ─────────────────────────────────────────────────
    all_model_names = {m["name"] for m in MODELS}
    invalid_models = [n for n in demo_model_names if n not in all_model_names]
    if invalid_models:
        print(f"ERROR: Unknown model(s): {invalid_models}. Valid: {sorted(all_model_names)}", file=sys.stderr)
        sys.exit(1)

    # ── Validate benchmark IDs ───────────────────────────────────────────────
    from mobile_llm_benchmark.config import BENCHMARK_BY_ID
    invalid_benches = [b for b in demo_bench_ids if b not in BENCHMARK_BY_ID]
    if invalid_benches:
        print(f"ERROR: Unknown benchmark(s): {invalid_benches}. Valid: {list(BENCHMARK_BY_ID)}", file=sys.stderr)
        sys.exit(1)

    demo_models = [m for m in MODELS if m["name"] in demo_model_names]

    # ── Header ───────────────────────────────────────────────────────────────
    mode_display = "MOCK (no API key)" if mock_mode else f"LIVE — {args.provider.upper()}"
    if rich_available:
        console = Console()
        console.print(Panel.fit(
            "[bold purple]Mobile LLM Benchmark Suite[/bold purple]\n"
            f"[dim]Mode: {'[yellow]' + mode_display + '[/yellow]' if mock_mode else '[green]' + mode_display + '[/green]'}"
            f" | Samples: {n_samples} | Output: {output_dir}{scenario_note}[/dim]",
            border_style="purple",
        ))
    else:
        print("=" * 60)
        print("Mobile LLM Benchmark Suite")
        print(f"Mode: {mode_display} | Samples: {n_samples}{scenario_note}")
        print("=" * 60)

    if rich_available:
        console.print(f"\n[bold]Models:[/bold] {', '.join(m['name'] for m in demo_models)}")
        console.print(f"[bold]Benchmarks:[/bold] {', '.join(demo_bench_ids)}")
    else:
        print(f"Models: {', '.join(m['name'] for m in demo_models)}")
        print(f"Benchmarks: {', '.join(demo_bench_ids)}")

    # ── Dry run — print config and exit ─────────────────────────────────────
    if args.dry_run:
        total_calls = len(demo_models) * len(demo_bench_ids) * n_samples
        if rich_available:
            console.print(
                f"\n[yellow]--dry-run[/yellow]: would make [bold]{total_calls:,}[/bold] API calls "
                f"({len(demo_models)} models × {len(demo_bench_ids)} benchmarks × {n_samples} samples). "
                "No calls made."
            )
        else:
            print(f"\n--dry-run: would make {total_calls:,} API calls. No calls made.")
        return

    # ── Run benchmarks ───────────────────────────────────────────────────────
    from mobile_llm_benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(mock_mode=mock_mode)
    results = []
    total = len(demo_models) * len(demo_bench_ids)

    if rich_available:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Benchmarking...", total=total)
            for model_cfg in demo_models:
                for bench_id in demo_bench_ids:
                    progress.update(
                        task,
                        description=f"[cyan]{model_cfg['name']}[/cyan] → [yellow]{bench_id}[/yellow]",
                    )
                    result = runner.run_single(model_cfg, bench_id, n_samples)
                    results.append(result)
                    progress.advance(task)
    else:
        for i, model_cfg in enumerate(demo_models):
            for j, bench_id in enumerate(demo_bench_ids):
                done = i * len(demo_bench_ids) + j + 1
                print(f"[{done}/{total}] {model_cfg['name']} → {bench_id}...", end=" ", flush=True)
                result = runner.run_single(model_cfg, bench_id, n_samples)
                results.append(result)
                print(f"{result.accuracy:.1%}")

    # ── Generate reports ─────────────────────────────────────────────────────
    if not args.no_report:
        from mobile_llm_benchmark.report_generator import ReportGenerator
        rg = ReportGenerator(output_dir=output_dir)
        paths = rg.generate_all(results)
    else:
        paths = {}

    # ── Print results table ──────────────────────────────────────────────────
    if rich_available:
        console.print("\n[bold green]Results[/bold green]")
        tbl = Table(show_header=True, header_style="bold purple")
        tbl.add_column("Model", style="cyan", width=18)
        tbl.add_column("Benchmark", style="yellow", width=14)
        tbl.add_column("Accuracy", justify="right", width=10)
        tbl.add_column("95% CI", style="dim", width=16)
        tbl.add_column("N", justify="right", width=6)
        tbl.add_column("Tokens", justify="right", width=8)

        for r in results:
            acc_color = "green" if r.accuracy >= 0.6 else ("yellow" if r.accuracy >= 0.4 else "red")
            tok_str = f"{r.tokens_total:,}" if r.tokens_total > 0 else "—"
            tbl.add_row(
                r.model_name,
                r.benchmark_name,
                f"[{acc_color}]{r.accuracy:.1%}[/{acc_color}]",
                f"[{r.ci_lower:.1%}, {r.ci_upper:.1%}]",
                str(r.n_samples),
                tok_str,
            )
        console.print(tbl)

        # Token summary
        total_tokens = sum(r.tokens_total for r in results)
        if total_tokens > 0:
            console.print(f"\n[dim]Total tokens used: {total_tokens:,}[/dim]")
    else:
        print("\nResults:")
        print(f"{'Model':<20} {'Benchmark':<16} {'Accuracy':>10} {'95% CI':>22} {'N':>6} {'Tokens':>10}")
        print("-" * 90)
        for r in results:
            tok_str = f"{r.tokens_total:,}" if r.tokens_total > 0 else "—"
            print(
                f"{r.model_name:<20} {r.benchmark_name:<16} {r.accuracy:>10.1%} "
                f"[{r.ci_lower:.1%}, {r.ci_upper:.1%}] {r.n_samples:>6} {tok_str:>10}"
            )

    # ── Print output paths ───────────────────────────────────────────────────
    if paths:
        if rich_available:
            console.print("\n[bold]Output files:[/bold]")
            for name, path in paths.items():
                console.print(f"  [dim]{name}:[/dim] [blue]{path}[/blue]")
        else:
            print("\nOutput files:")
            for name, path in paths.items():
                print(f"  {name}: {path}")

    if mock_mode:
        note = "⚠  Mock data — set OPENROUTER_API_KEY or HUGGINGFACE_TOKEN for real results"
        if rich_available:
            console.print(f"\n[yellow]{note}[/yellow]")
        else:
            print(f"\n{note}")

    print()


if __name__ == "__main__":
    main()
