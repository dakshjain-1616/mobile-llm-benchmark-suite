"""Report generator: HTML leaderboard, radar chart, bar chart, timing chart, PDF, JSON."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from jinja2 import Environment, BaseLoader

from .runner import BenchmarkResult
from .statistical import aggregate_scores, pairwise_effects

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template (inline Jinja2)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mobile LLM Benchmark Suite — Results</title>
  <style>
    :root {
      --bg: #0f1117; --surface: #1a1d27; --border: #2e3248;
      --text: #e2e8f0; --muted: #94a3b8; --accent: #7c3aed;
      --green: #10b981; --yellow: #f59e0b; --red: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
    .container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
    header { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg,#7c3aed,#2563eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .meta { color: var(--muted); font-size: .875rem; margin-top: .5rem; }
    .badge { display:inline-block; background:var(--accent); color:#fff; font-size:.7rem; padding:.15rem .5rem; border-radius:999px; margin-left:.5rem; vertical-align:middle; }
    .badge.mock { background: var(--yellow); color: #000; }
    h2 { font-size: 1.4rem; font-weight: 600; margin: 2rem 0 1rem; color: var(--text); }
    h3 { font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 .75rem; color: var(--muted); }
    table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }
    thead { background: rgba(124,58,237,.15); }
    th { padding: .75rem 1rem; text-align: left; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); cursor: pointer; user-select: none; }
    th:hover { color: var(--text); }
    th.sorted-asc::after { content: " ▲"; }
    th.sorted-desc::after { content: " ▼"; }
    td { padding: .65rem 1rem; border-top: 1px solid var(--border); font-size: .9rem; }
    tr:hover td { background: rgba(124,58,237,.07); }
    .rank { font-weight: 700; color: var(--accent); width: 2.5rem; }
    .acc { font-weight: 600; }
    .acc.high { color: var(--green); }
    .acc.mid { color: var(--yellow); }
    .acc.low { color: var(--red); }
    .ci { font-size: .75rem; color: var(--muted); }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; margin: 1.5rem 0; }
    .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; text-align: center; }
    .chart-card img { max-width: 100%; border-radius: 4px; }
    .methodology { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-top: 2rem; }
    .methodology p, .methodology li { color: var(--muted); font-size: .9rem; line-height: 1.7; }
    .methodology ul { padding-left: 1.5rem; margin-top: .5rem; }
    .params { font-size: .75rem; color: var(--muted); }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0 2rem; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
    .stat-label { font-size: .75rem; color: var(--muted); margin-top: .25rem; }
    footer { margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--muted); font-size: .8rem; text-align: center; }
    footer a { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>Mobile LLM Benchmark Suite</h1>
    <p class="meta">
      Generated: {{ timestamp }} &nbsp;|&nbsp;
      Models: {{ n_models }} &nbsp;|&nbsp;
      Benchmarks: {{ n_benchmarks }} &nbsp;|&nbsp;
      Samples/bench: {{ n_samples }}
      {% if mock %}<span class="badge mock">MOCK DATA</span>{% else %}<span class="badge">LIVE DATA</span>{% endif %}
    </p>
  </header>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value">{{ n_models }}</div>
      <div class="stat-label">Models Evaluated</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{{ n_benchmarks }}</div>
      <div class="stat-label">Benchmarks</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{{ total_samples }}</div>
      <div class="stat-label">Total Evaluations</div>
    </div>
    {% if total_tokens > 0 %}
    <div class="stat-card">
      <div class="stat-value">{{ "{:,}".format(total_tokens) }}</div>
      <div class="stat-label">Total Tokens Used</div>
    </div>
    {% endif %}
    <div class="stat-card">
      <div class="stat-value">{{ best_model }}</div>
      <div class="stat-label">Top Model</div>
    </div>
  </div>

  <h2>🏆 Leaderboard</h2>
  <table id="leaderboard">
    <thead>
      <tr>
        <th class="rank">Rank</th>
        <th data-col="model">Model</th>
        <th data-col="params">Params</th>
        <th data-col="avg" class="sorted-desc">Avg Accuracy</th>
        {% for b in benchmark_names %}<th data-col="{{ b }}">{{ b }}</th>{% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for row in leaderboard %}
      <tr>
        <td class="rank">{{ row.rank }}</td>
        <td><strong>{{ row.model_name }}</strong></td>
        <td class="params">{{ row.params }}</td>
        <td class="acc {% if row.avg_accuracy >= 0.6 %}high{% elif row.avg_accuracy >= 0.4 %}mid{% else %}low{% endif %}">
          {{ "%.1f"|format(row.avg_accuracy * 100) }}%
        </td>
        {% for b_id in benchmark_ids %}
        {% set cell = row.cells.get(b_id) %}
        {% if cell %}
        <td class="acc {% if cell.accuracy >= 0.6 %}high{% elif cell.accuracy >= 0.4 %}mid{% else %}low{% endif %}">
          {{ "%.1f"|format(cell.accuracy * 100) }}%
          <span class="ci">[{{ "%.0f"|format(cell.ci_lower*100) }}–{{ "%.0f"|format(cell.ci_upper*100) }}]</span>
        </td>
        {% else %}
        <td class="muted">—</td>
        {% endif %}
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>📊 Visualizations</h2>
  <div class="chart-grid">
    {% if radar_b64 %}
    <div class="chart-card">
      <h3>Radar Chart — Per-Model Profile</h3>
      <img src="data:image/png;base64,{{ radar_b64 }}" alt="Radar chart" />
    </div>
    {% endif %}
    {% if bar_b64 %}
    <div class="chart-card">
      <h3>Per-Benchmark Scores</h3>
      <img src="data:image/png;base64,{{ bar_b64 }}" alt="Bar chart" />
    </div>
    {% endif %}
    {% if timing_b64 %}
    <div class="chart-card">
      <h3>Avg Latency per Model (ms)</h3>
      <img src="data:image/png;base64,{{ timing_b64 }}" alt="Timing chart" />
    </div>
    {% endif %}
  </div>

  {% if mmlu_category_data %}
  <h2>📚 MMLU Category Breakdown</h2>
  <table>
    <thead>
      <tr><th>Model</th>{% for cat in mmlu_categories %}<th>{{ cat }}</th>{% endfor %}</tr>
    </thead>
    <tbody>
      {% for model_name, cat_scores in mmlu_category_data.items() %}
      <tr>
        <td><strong>{{ model_name }}</strong></td>
        {% for cat in mmlu_categories %}
        {% set sc = cat_scores.get(cat) %}
        {% if sc is not none %}
        <td class="acc {% if sc >= 0.6 %}high{% elif sc >= 0.4 %}mid{% else %}low{% endif %}">
          {{ "%.1f"|format(sc * 100) }}%
        </td>
        {% else %}
        <td>—</td>
        {% endif %}
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <h2>📋 Detailed Results</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th><th>Benchmark</th><th>Accuracy</th>
        <th>95% CI Lower</th><th>95% CI Upper</th>
        <th>N Correct</th><th>N Samples</th>
        <th>Tokens</th><th>Avg Latency</th>
      </tr>
    </thead>
    <tbody>
      {% for r in all_results %}
      <tr>
        <td>{{ r.model_name }}</td>
        <td>{{ r.benchmark_name }}</td>
        <td class="acc {% if r.accuracy >= 0.6 %}high{% elif r.accuracy >= 0.4 %}mid{% else %}low{% endif %}">
          {{ "%.2f"|format(r.accuracy * 100) }}%
        </td>
        <td class="ci">{{ "%.2f"|format(r.ci_lower * 100) }}%</td>
        <td class="ci">{{ "%.2f"|format(r.ci_upper * 100) }}%</td>
        <td>{{ r.n_correct }}</td>
        <td>{{ r.n_samples }}</td>
        <td class="ci">{% if r.tokens_total > 0 %}{{ "{:,}".format(r.tokens_total) }}{% else %}—{% endif %}</td>
        <td class="ci">{% if r.avg_latency_ms > 0 %}{{ "%.0f"|format(r.avg_latency_ms) }}ms{% else %}—{% endif %}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <div class="methodology">
    <h2>📐 Methodology</h2>
    <p>
      All models are evaluated in zero-shot setting (no in-context examples) using
      the OpenRouter or HuggingFace Inference API — no local GPU required.
      Temperature is set to 0 for reproducibility.
    </p>
    <h3>Benchmarks</h3>
    <ul>
      <li><strong>GSM8K</strong> — Grade-school math word problems. Metric: exact number match after chain-of-thought.</li>
      <li><strong>ARC-Challenge</strong> — AI2 hard science multiple-choice. Metric: letter extraction (A–D).</li>
      <li><strong>MMLU</strong> — 57-subject multiple-choice. Metric: letter extraction (A–D).</li>
      <li><strong>HellaSwag</strong> — Commonsense NLI sentence completion. Metric: choice index (0–3).</li>
      <li><strong>TruthfulQA</strong> — Truthfulness multiple-choice. Metric: letter extraction (A–D).</li>
      <li><strong>IFEval</strong> — Instruction following (format/keyword/length constraints). Metric: per-instruction binary check, strict AND across all instructions.</li>
    </ul>
    <h3>Statistics</h3>
    <p>
      Confidence intervals are Wilson score intervals at 95% confidence level.
      The Wilson interval has better coverage than the Wald interval for small n and
      extreme proportions. Effect sizes between model pairs use Cohen's h for proportions
      (h &lt; 0.2 negligible, h &lt; 0.5 small, h &lt; 0.8 medium, h ≥ 0.8 large).
      Pairwise significance uses a two-proportion z-test (α=0.05).
    </p>
    <h3>Reproducibility</h3>
    <p>
      All datasets are loaded from HuggingFace Hub with a fixed random seed (42) for
      stratified sampling. Results are deterministic across runs with the same seed.
    </p>
  </div>

  <footer>
    <p>
      Generated by <a href="https://github.com/dakshjain-1616/mobile-llm-benchmark-suite">Mobile LLM Benchmark Suite</a>
      &nbsp;·&nbsp; Made with <a href="https://heyneo.so">NEO</a>
    </p>
  </footer>
</div>

<script>
// Sortable table
(function() {
  const table = document.getElementById('leaderboard');
  if (!table) return;
  const headers = table.querySelectorAll('th[data-col]');
  let sortCol = 'avg', sortDir = -1;

  function sortTable(col, dir) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const ai = getCell(a, col), bi = getCell(b, col);
      const av = parseFloat(ai) || ai || '', bv = parseFloat(bi) || bi || '';
      if (av < bv) return dir;
      if (av > bv) return -dir;
      return 0;
    });
    rows.forEach(r => tbody.appendChild(r));
    // Re-number rank
    rows.forEach((r, i) => { const td = r.querySelector('.rank'); if(td) td.textContent = i+1; });
  }

  function getCell(row, col) {
    const cells = row.querySelectorAll('td');
    const headerIdx = Array.from(headers).findIndex(h => h.dataset.col === col);
    if (headerIdx < 0) return '';
    const td = cells[headerIdx + 1]; // +1 for rank col
    return td ? td.textContent.replace('%','').replace(/\\[.*\\]/,'').trim() : '';
  }

  headers.forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) sortDir *= -1;
      else { sortCol = col; sortDir = -1; }
      headers.forEach(h => h.classList.remove('sorted-asc','sorted-desc'));
      th.classList.add(sortDir === -1 ? 'sorted-desc' : 'sorted-asc');
      sortTable(sortCol, sortDir);
    });
  });
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

_COLORS = [
    "#7c3aed", "#2563eb", "#10b981", "#f59e0b", "#ef4444",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#8b5cf6",
]


class ReportGenerator:
    """Generates HTML report, PNG charts, CSV, JSON, and PDF from benchmark results."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(os.getenv("OUTPUT_DIR", output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_all(
        self, results: list[BenchmarkResult]
    ) -> dict[str, Path]:
        """Generate all outputs and return mapping of name→path."""
        if not results:
            raise ValueError("No results to generate report from.")

        df = self.results_to_df(results)
        leaderboard = self._build_leaderboard(results)
        benchmark_names = sorted({r.benchmark_name for r in results})
        benchmark_ids = sorted({r.benchmark for r in results})

        radar_path = self.save_radar_chart(results)
        bar_path = self.save_bar_chart(results)
        timing_path = self.save_timing_chart(results)

        radar_b64 = _img_to_b64(radar_path)
        bar_b64 = _img_to_b64(bar_path)
        timing_b64 = _img_to_b64(timing_path) if timing_path and timing_path.exists() else ""

        # MMLU category breakdown
        mmlu_category_data, mmlu_categories = self._build_mmlu_category_breakdown(results)

        html_path = self._render_html(
            results, leaderboard, benchmark_names, benchmark_ids,
            radar_b64, bar_b64, timing_b64,
            mmlu_category_data, mmlu_categories,
        )
        csv_path = self.save_csv(df)
        json_path = self.save_json(results)
        leaderboard_path = self.save_leaderboard_html(leaderboard, benchmark_names, benchmark_ids)
        pdf_path = self.save_pdf(results, leaderboard, benchmark_names, benchmark_ids)

        output = {
            "report_html": html_path,
            "leaderboard_html": leaderboard_path,
            "radar_chart": radar_path,
            "bar_chart": bar_path,
            "csv": csv_path,
            "json": json_path,
            "pdf": pdf_path,
        }
        if timing_path and timing_path.exists():
            output["timing_chart"] = timing_path
        return output

    # ------------------------------------------------------------------
    # DataFrame / CSV
    # ------------------------------------------------------------------

    def results_to_df(self, results: list[BenchmarkResult]) -> pd.DataFrame:
        rows = [r.as_dict() for r in results]
        return pd.DataFrame(rows)

    def save_csv(self, df: pd.DataFrame) -> Path:
        path = self.output_dir / "benchmark_results.csv"
        for col in ["model", "benchmark", "accuracy", "ci_lower", "ci_upper", "n_samples"]:
            if col not in df.columns:
                df[col] = None
        df.to_csv(path, index=False)
        logger.info("Saved CSV → %s", path)
        return path

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def save_json(self, results: list[BenchmarkResult]) -> Path:
        """Export results + summary statistics as a structured JSON file."""
        path = self.output_dir / "benchmark_results.json"

        # Build per-model summary
        model_summaries: dict[str, dict] = {}
        for r in results:
            if r.model_name not in model_summaries:
                model_summaries[r.model_name] = {
                    "model_id": r.model_id,
                    "mock": r.mock,
                    "benchmarks": {},
                    "total_tokens": 0,
                }
            model_summaries[r.model_name]["benchmarks"][r.benchmark] = {
                "accuracy": round(r.accuracy, 6),
                "ci_lower": round(r.ci_lower, 6),
                "ci_upper": round(r.ci_upper, 6),
                "n_samples": r.n_samples,
                "n_correct": r.n_correct,
                "duration_s": round(r.duration_s, 2),
                "tokens_total": r.tokens_total,
                "avg_latency_ms": round(r.avg_latency_ms, 1),
                "p95_latency_ms": round(r.p95_latency_ms, 1),
            }
            model_summaries[r.model_name]["total_tokens"] += r.tokens_total

        # Compute per-model average accuracy
        for name, summary in model_summaries.items():
            accs = [v["accuracy"] for v in summary["benchmarks"].values()]
            summary["avg_accuracy"] = round(float(np.mean(accs)), 6) if accs else 0.0

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_models": len(model_summaries),
            "n_benchmarks": len({r.benchmark for r in results}),
            "mock": any(r.mock for r in results),
            "total_tokens": sum(r.tokens_total for r in results),
            "models": model_summaries,
            "raw_results": [r.as_dict() for r in results],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved JSON → %s", path)
        return path

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def _build_leaderboard(self, results: list[BenchmarkResult]) -> list[dict]:
        """Build sorted leaderboard rows."""
        from .config import MODELS

        model_params = {m["name"]: m["params"] for m in MODELS}

        idx: dict[tuple, BenchmarkResult] = {}
        for r in results:
            idx[(r.model_name, r.benchmark)] = r

        model_names = list(dict.fromkeys(r.model_name for r in results))

        rows = []
        for name in model_names:
            cells = {
                bench: idx[(name, bench)]
                for (n, bench) in idx
                if n == name
            }
            accs = [c.accuracy for c in cells.values()]
            avg = float(np.mean(accs)) if accs else 0.0
            rows.append({
                "model_name": name,
                "params": model_params.get(name, "?B"),
                "avg_accuracy": avg,
                "cells": cells,
            })

        rows.sort(key=lambda r: r["avg_accuracy"], reverse=True)
        for i, row in enumerate(rows):
            row["rank"] = i + 1
        return rows

    # ------------------------------------------------------------------
    # MMLU category breakdown
    # ------------------------------------------------------------------

    def _build_mmlu_category_breakdown(
        self, results: list[BenchmarkResult]
    ) -> tuple[dict[str, dict[str, Optional[float]]], list[str]]:
        """Extract per-subject MMLU scores from category_scores field."""
        mmlu_results = [r for r in results if r.benchmark == "mmlu" and r.category_scores]
        if not mmlu_results:
            return {}, []

        all_cats: set[str] = set()
        for r in mmlu_results:
            all_cats.update(r.category_scores.keys())
        categories = sorted(all_cats)[:10]  # Cap at 10 for readability

        data: dict[str, dict[str, Optional[float]]] = {}
        for r in mmlu_results:
            data[r.model_name] = {cat: r.category_scores.get(cat) for cat in categories}

        return data, categories

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _render_html(
        self,
        results: list[BenchmarkResult],
        leaderboard: list[dict],
        benchmark_names: list[str],
        benchmark_ids: list[str],
        radar_b64: str,
        bar_b64: str,
        timing_b64: str,
        mmlu_category_data: dict,
        mmlu_categories: list[str],
    ) -> Path:
        env = Environment(loader=BaseLoader())
        template = env.from_string(_HTML_TEMPLATE)
        mock = any(r.mock for r in results)
        n_samples_set = list({r.n_samples for r in results})
        n_samples = n_samples_set[0] if len(n_samples_set) == 1 else f"~{int(np.mean(n_samples_set))}"
        total_tokens = sum(r.tokens_total for r in results)
        best_model = leaderboard[0]["model_name"] if leaderboard else "—"

        html = template.render(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            n_models=len({r.model_name for r in results}),
            n_benchmarks=len({r.benchmark for r in results}),
            n_samples=n_samples,
            total_samples=sum(r.n_samples for r in results),
            total_tokens=total_tokens,
            best_model=best_model,
            mock=mock,
            leaderboard=leaderboard,
            benchmark_names=benchmark_names,
            benchmark_ids=benchmark_ids,
            all_results=results,
            radar_b64=radar_b64,
            bar_b64=bar_b64,
            timing_b64=timing_b64,
            mmlu_category_data=mmlu_category_data,
            mmlu_categories=mmlu_categories,
        )
        path = self.output_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved report → %s", path)
        return path

    def save_leaderboard_html(
        self,
        leaderboard: list[dict],
        benchmark_names: list[str],
        benchmark_ids: list[str],
    ) -> Path:
        """Standalone sortable leaderboard HTML file."""
        rows_html = ""
        for row in leaderboard:
            cells_html = ""
            for b_id in benchmark_ids:
                cell = row["cells"].get(b_id)
                if cell:
                    cls = "high" if cell.accuracy >= 0.6 else ("mid" if cell.accuracy >= 0.4 else "low")
                    cells_html += (
                        f'<td class="acc {cls}">{cell.accuracy:.1%} '
                        f'<span class="ci">[{cell.ci_lower:.0%}–{cell.ci_upper:.0%}]</span></td>'
                    )
                else:
                    cells_html += "<td>—</td>"

            cls = "high" if row["avg_accuracy"] >= 0.6 else ("mid" if row["avg_accuracy"] >= 0.4 else "low")
            rows_html += (
                f'<tr><td class="rank">{row["rank"]}</td>'
                f'<td><strong>{row["model_name"]}</strong></td>'
                f'<td class="params">{row["params"]}</td>'
                f'<td class="acc {cls}">{row["avg_accuracy"]:.1%}</td>'
                f"{cells_html}</tr>\n"
            )

        bench_headers = "".join(f"<th>{b}</th>" for b in benchmark_names)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Mobile LLM Leaderboard</title>
  <style>
    body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0;padding:2rem}}
    table{{border-collapse:collapse;width:100%;background:#1a1d27;border-radius:8px;overflow:hidden}}
    th{{padding:.75rem 1rem;text-align:left;background:rgba(124,58,237,.2);cursor:pointer;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8}}
    td{{padding:.65rem 1rem;border-top:1px solid #2e3248;font-size:.9rem}}
    tr:hover td{{background:rgba(124,58,237,.07)}}
    .rank{{color:#7c3aed;font-weight:700}}.params{{color:#94a3b8;font-size:.75rem}}
    .high{{color:#10b981;font-weight:600}}.mid{{color:#f59e0b;font-weight:600}}.low{{color:#ef4444;font-weight:600}}
    .ci{{font-size:.72rem;color:#94a3b8}}
    h1{{background:linear-gradient(135deg,#7c3aed,#2563eb);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.8rem}}
  </style>
</head>
<body>
  <h1>Mobile LLM Benchmark Leaderboard</h1>
  <p style="color:#94a3b8;margin:.5rem 0 1.5rem">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
  <table id="lb">
    <thead>
      <tr>
        <th>Rank</th><th>Model</th><th>Params</th><th>Avg Accuracy</th>
        {bench_headers}
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  <script>
  const t=document.getElementById('lb');
  let sd=-1,sc=3;
  t.querySelectorAll('th').forEach((th,i)=>{{
    th.addEventListener('click',()=>{{
      if(sc===i)sd*=-1;else{{sc=i;sd=-1;}}
      const rows=[...t.querySelectorAll('tbody tr')];
      rows.sort((a,b)=>{{
        const av=a.cells[i].textContent.replace(/[^\\d.]/g,''),bv=b.cells[i].textContent.replace(/[^\\d.]/g,'');
        return (parseFloat(av)||av)<(parseFloat(bv)||bv)?sd:-sd;
      }});
      rows.forEach(r=>t.querySelector('tbody').appendChild(r));
      rows.forEach((r,i)=>r.cells[0].textContent=i+1);
    }});
  }});
  </script>
</body>
</html>"""
        path = self.output_dir / "leaderboard.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved leaderboard → %s", path)
        return path

    # ------------------------------------------------------------------
    # Radar chart
    # ------------------------------------------------------------------

    def save_radar_chart(self, results: list[BenchmarkResult]) -> Path:
        benchmarks = sorted({r.benchmark for r in results})
        models = list(dict.fromkeys(r.model_name for r in results))

        if len(benchmarks) < 3:
            benchmarks += [f"pad_{i}" for i in range(3 - len(benchmarks))]

        n_axes = len(benchmarks)
        angles = [2 * np.pi * i / n_axes for i in range(n_axes)]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        ax.set_facecolor("#1a1d27")
        fig.patch.set_facecolor("#0f1117")

        ax.set_thetagrids(
            [a * 180 / np.pi for a in angles],
            labels=[b.upper() for b in benchmarks],
            fontsize=9, color="#94a3b8",
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=7, color="#94a3b8")
        ax.grid(color="#2e3248", linestyle="--", alpha=0.6)
        ax.spines["polar"].set_color("#2e3248")

        idx: dict[tuple, float] = {(r.model_name, r.benchmark): r.accuracy for r in results}

        for i, model in enumerate(models[:10]):
            color = _COLORS[i % len(_COLORS)]
            values = [idx.get((model, b), 0.0) for b in benchmarks]
            values_closed = values + [values[0]]
            ax.plot(angles_closed, values_closed, "o-", color=color, linewidth=2, markersize=4, label=model)
            ax.fill(angles_closed, values_closed, color=color, alpha=0.08)

        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.35, -0.05),
            fontsize=8,
            labelcolor="#e2e8f0",
            framealpha=0.0,
        )
        ax.set_title("Model Performance Radar", color="#e2e8f0", fontsize=13, pad=20)

        path = self.output_dir / "radar_chart.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        logger.info("Saved radar chart → %s", path)
        return path

    # ------------------------------------------------------------------
    # Bar chart
    # ------------------------------------------------------------------

    def save_bar_chart(self, results: list[BenchmarkResult]) -> Path:
        benchmarks = sorted({r.benchmark for r in results})
        models = list(dict.fromkeys(r.model_name for r in results))

        idx: dict[tuple, BenchmarkResult] = {
            (r.model_name, r.benchmark): r for r in results
        }

        n_benchmarks = len(benchmarks)
        n_models = len(models)
        bar_width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=(max(10, n_benchmarks * 2), 6))
        ax.set_facecolor("#1a1d27")
        fig.patch.set_facecolor("#0f1117")

        x = np.arange(n_benchmarks)

        for i, model in enumerate(models[:10]):
            color = _COLORS[i % len(_COLORS)]
            accs = [idx.get((model, b), None) for b in benchmarks]
            heights = [r.accuracy if r else 0 for r in accs]
            ci_lows = [r.accuracy - r.ci_lower if r else 0 for r in accs]
            ci_highs = [r.ci_upper - r.accuracy if r else 0 for r in accs]
            offset = (i - n_models / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, heights, bar_width * 0.9,
                label=model, color=color, alpha=0.85,
                yerr=[ci_lows, ci_highs],
                error_kw={"ecolor": "white", "capsize": 3, "alpha": 0.6, "linewidth": 1},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([b.upper() for b in benchmarks], color="#94a3b8", fontsize=10)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color="#94a3b8")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-Benchmark Accuracy (with 95% CI)", color="#e2e8f0", fontsize=13)
        ax.set_ylabel("Accuracy", color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#2e3248")
        ax.tick_params(colors="#94a3b8")
        ax.grid(axis="y", color="#2e3248", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, labelcolor="#e2e8f0", framealpha=0.0, loc="upper right")

        path = self.output_dir / "bar_chart.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        logger.info("Saved bar chart → %s", path)
        return path

    # ------------------------------------------------------------------
    # Timing chart (new)
    # ------------------------------------------------------------------

    def save_timing_chart(self, results: list[BenchmarkResult]) -> Optional[Path]:
        """Horizontal bar chart of average latency per model.

        Returns None and skips if no latency data is available.
        """
        # Aggregate avg latency per model
        model_latencies: dict[str, list[float]] = defaultdict(list)
        model_p95: dict[str, list[float]] = defaultdict(list)
        for r in results:
            if r.avg_latency_ms > 0:
                model_latencies[r.model_name].append(r.avg_latency_ms)
            if r.p95_latency_ms > 0:
                model_p95[r.model_name].append(r.p95_latency_ms)

        if not model_latencies:
            logger.debug("No latency data — skipping timing chart")
            return None

        models = list(model_latencies.keys())
        avg_lats = [float(np.mean(model_latencies[m])) for m in models]
        p95_lats = [float(np.mean(model_p95.get(m, [avg_lats[i]]))) for i, m in enumerate(models)]

        # Sort by avg latency
        sorted_pairs = sorted(zip(avg_lats, p95_lats, models))
        avg_lats, p95_lats, models = (
            [x[0] for x in sorted_pairs],
            [x[1] for x in sorted_pairs],
            [x[2] for x in sorted_pairs],
        )

        fig, ax = plt.subplots(figsize=(9, max(4, len(models) * 0.55)))
        ax.set_facecolor("#1a1d27")
        fig.patch.set_facecolor("#0f1117")

        y = np.arange(len(models))
        ax.barh(y, avg_lats, color="#7c3aed", alpha=0.85, label="Avg latency")
        ax.barh(y, p95_lats, color="#2563eb", alpha=0.4, label="p95 latency")

        ax.set_yticks(y)
        ax.set_yticklabels(models, color="#e2e8f0", fontsize=9)
        ax.set_xlabel("Latency (ms)", color="#94a3b8")
        ax.set_title("Model Response Latency", color="#e2e8f0", fontsize=12)
        for spine in ax.spines.values():
            spine.set_color("#2e3248")
        ax.tick_params(colors="#94a3b8")
        ax.grid(axis="x", color="#2e3248", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, labelcolor="#e2e8f0", framealpha=0.0)

        path = self.output_dir / "timing_chart.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        logger.info("Saved timing chart → %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------

    def save_pdf(
        self,
        results: list[BenchmarkResult],
        leaderboard: list[dict],
        benchmark_names: list[str],
        benchmark_ids: list[str],
    ) -> Path:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, Image as RLImage,
            )
            from reportlab.lib.enums import TA_CENTER
        except ImportError:
            logger.warning("reportlab not available — skipping PDF generation")
            return self.output_dir / "report.pdf"

        path = self.output_dir / "report.pdf"
        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title2",
            parent=styles["Title"],
            textColor=colors.HexColor("#7c3aed"),
            fontSize=20,
            spaceAfter=6,
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            textColor=colors.HexColor("#1e293b"),
            fontSize=13,
            spaceBefore=14, spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body2", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        )
        meta_style = ParagraphStyle(
            "Meta", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#64748b"),
        )

        story = []
        story.append(Paragraph("Mobile LLM Benchmark Suite", title_style))
        mock = any(r.mock for r in results)
        total_tokens = sum(r.tokens_total for r in results)
        token_note = f" | Tokens: {total_tokens:,}" if total_tokens > 0 else ""
        story.append(Paragraph(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
            f"| Models: {len({r.model_name for r in results})} "
            f"| {'Mock Data' if mock else 'Live Data'}{token_note}",
            meta_style,
        ))
        story.append(Spacer(1, 0.2 * inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 0.15 * inch))

        # Leaderboard table
        story.append(Paragraph("Leaderboard", h2_style))
        header_row = ["Rank", "Model", "Params", "Avg Acc"] + benchmark_names
        table_data = [header_row]
        for row in leaderboard:
            cells = []
            for b_id in benchmark_ids:
                cell = row["cells"].get(b_id)
                cells.append(f"{cell.accuracy:.1%}" if cell else "—")
            table_data.append([
                str(row["rank"]),
                row["model_name"],
                row["params"],
                f"{row['avg_accuracy']:.1%}",
                *cells,
            ])

        col_widths = [0.5 * inch, 1.5 * inch, 0.6 * inch, 0.7 * inch]
        col_widths += [0.8 * inch] * len(benchmark_names)

        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7c3aed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))

        # Charts
        for chart_name in ["radar_chart.png", "bar_chart.png", "timing_chart.png"]:
            chart_path = self.output_dir / chart_name
            if chart_path.exists():
                story.append(Paragraph(
                    chart_name.replace("_", " ").replace(".png", "").title(),
                    h2_style,
                ))
                try:
                    img = RLImage(str(chart_path), width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 0.1 * inch))
                except Exception:
                    pass

        # Methodology
        story.append(Paragraph("Methodology", h2_style))
        story.append(Paragraph(
            "All models evaluated zero-shot via OpenRouter or HuggingFace Inference API. "
            "Confidence intervals use Wilson score interval at 95% confidence. "
            "Effect sizes use Cohen's h for proportions. "
            "Pairwise significance uses two-proportion z-test (α=0.05).",
            body_style,
        ))

        doc.build(story)
        logger.info("Saved PDF → %s", path)
        return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _img_to_b64(path: Optional[Path]) -> str:
    """Read a PNG file and return its base64-encoded string."""
    if path is None or not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")
