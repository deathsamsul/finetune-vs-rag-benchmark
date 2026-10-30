#!/usr/bin/env python3
"""
visualize_benchmark.py
Create publication-ready charts, an HTML report, and a social-post draft
from evaluation/results/ragas_scores.json and disagreements.csv.

Run from the repository root:
    python evaluation/visualize_benchmark.py

Optional:
    python evaluation/visualize_benchmark.py --results-dir evaluation/results
    python evaluation/visualize_benchmark.py --output-dir evaluation/results/visualizations
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# python -m evaluation.visualize_benchmark

DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_OUTPUT_DIR = Path("evaluation/results/visualizations")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize fine-tuned vs RAG benchmark results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"WARNING: {path} not found. CSV-based charts will be skipped.")
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean_numeric(rows: list[dict[str, str]], key: str) -> float | None:
    values = [v for row in rows if (v := number(row.get(key))) is not None]
    return statistics.mean(values) if values else None


def save_metric_comparison(scores: dict, out: Path) -> bool:
    ft = scores.get("finetuned", {})
    rag = scores.get("rag", {})
    metrics = list(dict.fromkeys([*ft.keys(), *rag.keys()]))
    if not metrics:
        return False

    x = list(range(len(metrics)))
    width = 0.36
    ft_values = [number(ft.get(m)) for m in metrics]
    rag_values = [number(rag.get(m)) for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ft_plot = [v if v is not None else 0 for v in ft_values]
    rag_plot = [v if v is not None else 0 for v in rag_values]
    ax.bar([i - width / 2 for i in x], ft_plot, width, label="Fine-tuned")
    ax.bar([i + width / 2 for i in x], rag_plot, width, label="RAG")

    for i, val in enumerate(ft_values):
        if val is not None:
            ax.text(i - width / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for i, val in enumerate(rag_values):
        if val is not None:
            ax.text(i + width / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, [m.replace("_", "\n") for m in metrics])
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Fine-tuned vs RAG — Aggregate Evaluation Metrics")
    ax.set_ylim(0, max(1.12, max([v for v in ft_values + rag_values if v is not None] + [1]) + 0.1))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "aggregate_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_latency_chart(rows: list[dict[str, str]], out: Path) -> dict[str, float] | None:
    ft = mean_numeric(rows, "ft_latency_ms")
    rag = mean_numeric(rows, "rag_latency_ms")
    if ft is None and rag is None:
        return None

    labels, values = [], []
    if ft is not None:
        labels.append("Fine-tuned")
        values.append(ft / 1000)
    if rag is not None:
        labels.append("RAG")
        values.append(rag / 1000)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, values)
    ax.bar_label(bars, labels=[f"{v:.2f} s" for v in values], padding=4)
    ax.set_ylabel("Average recorded latency (seconds)")
    ax.set_title("Inference Latency Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    fig.tight_layout()
    fig.savefig(out / "latency_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    result = {}
    if ft is not None:
        result["finetuned_seconds"] = ft / 1000
    if rag is not None:
        result["rag_seconds"] = rag / 1000
    return result


def save_per_sample_chart(rows: list[dict[str, str]], out: Path) -> str | None:
    # Support both the older one-metric CSV and the newer two-metric CSV.
    pairs = [
        ("ft_faithfulness", "rag_faithfulness", "Faithfulness"),
        ("ft_answer_relevancy", "rag_answer_relevancy", "Answer relevancy"),
        ("ft_score", "rag_score", "Score (metric selected during run)"),
    ]
    selected = next(
        ((a, b, title) for a, b, title in pairs
         if any(number(r.get(a)) is not None and number(r.get(b)) is not None for r in rows)),
        None,
    )
    if not selected:
        return None

    ft_key, rag_key, title = selected
    usable = [
        (i + 1, number(row.get(ft_key)), number(row.get(rag_key)))
        for i, row in enumerate(rows)
        if number(row.get(ft_key)) is not None and number(row.get(rag_key)) is not None
    ]
    if not usable:
        return None

    labels = [str(item[0]) for item in usable]
    ft_values = [item[1] for item in usable]
    rag_values = [item[2] for item in usable]
    x = list(range(len(usable)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(usable) * 0.8), 5.3))
    ax.bar([i - width / 2 for i in x], ft_values, width, label="Fine-tuned")
    ax.bar([i + width / 2 for i in x], rag_values, width, label="RAG")
    ax.set_xticks(x, labels)
    ax.set_xlabel("CSV row (rank/order)")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-sample comparison — {title}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "per_sample_score_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return title


def save_delta_chart(rows: list[dict[str, str]], out: Path) -> str | None:
    delta_keys = [
        ("faithfulness_delta", "Faithfulness"),
        ("answer_relevancy_delta", "Answer relevancy"),
        ("delta", "Selected metric"),
    ]
    selected = next(
        ((key, title) for key, title in delta_keys
         if any(number(r.get(key)) is not None for r in rows)),
        None,
    )
    if not selected:
        return None
    key, title = selected
    usable = [
        (i + 1, number(row.get(key)))
        for i, row in enumerate(rows)
        if number(row.get(key)) is not None
    ]
    if not usable:
        return None

    # Show up to 30 rows so labels remain legible.
    usable = usable[:30]
    labels = [str(item[0]) for item in usable]
    values = [item[1] for item in usable]
    fig, ax = plt.subplots(figsize=(max(8, len(usable) * 0.45), 5.3))
    ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("CSV row (rank/order)")
    ax.set_ylabel("Fine-tuned score − RAG score")
    ax.set_title(f"Per-sample score difference — {title}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "score_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return title


def fmt(value: Any, digits: int = 3) -> str:
    val = number(value)
    return "—" if val is None else f"{val:.{digits}f}"


def main():
    args = parse_args()
    results_dir = args.results_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    scores_path = results_dir / "ragas_scores.json"
    csv_path = results_dir / "disagreements.csv"
    data = read_json(scores_path)
    rows = read_csv_rows(csv_path)
    scores = data.get("scores", {})
    ft = scores.get("finetuned", {})
    rag = scores.get("rag", {})
    warnings = []

    declared_n = data.get("n_samples")
    if declared_n is not None and rows and int(declared_n) != len(rows):
        warnings.append(
            f"Aggregate JSON says n_samples={declared_n}, but disagreements.csv contains {len(rows)} row(s). "
            "These files may come from different runs or the CSV may contain only a subset."
        )
    if not rows:
        warnings.append("No disagreement rows were loaded; per-question and latency charts may be unavailable.")

    chart_files = []
    if save_metric_comparison(scores, out):
        chart_files.append(("Aggregate metric comparison", "aggregate_metric_comparison.png"))

    latency = save_latency_chart(rows, out)
    if latency is not None:
        chart_files.append(("Inference latency comparison", "latency_comparison.png"))

    sample_title = save_per_sample_chart(rows, out)
    if sample_title:
        chart_files.append((f"Per-sample scores — {sample_title}", "per_sample_score_comparison.png"))

    delta_title = save_delta_chart(rows, out)
    if delta_title:
        chart_files.append((f"Per-sample score differences — {delta_title}", "score_delta.png"))

    # Compact metric table and evidence-based observations.
    all_metrics = list(dict.fromkeys([*ft.keys(), *rag.keys()]))
    table_rows = []
    for metric in all_metrics:
        fval = number(ft.get(metric))
        rval = number(rag.get(metric))
        if fval is not None and rval is not None:
            winner = "Fine-tuned" if fval > rval else ("RAG" if rval > fval else "Tie")
        elif fval is not None:
            winner = "Fine-tuned only"
        elif rval is not None:
            winner = "RAG only"
        else:
            winner = "—"
        table_rows.append((metric, fmt(fval), fmt(rval), winner))

    observations = []
    for metric in all_metrics:
        fval, rval = number(ft.get(metric)), number(rag.get(metric))
        if fval is not None and rval is not None:
            difference = fval - rval
            leader = "fine-tuned" if difference > 0 else ("RAG" if difference < 0 else "neither")
            observations.append(
                f"For **{metric.replace('_', ' ')}**, {leader} has the higher aggregate score "
                f"(fine-tuned {fval:.3f}; RAG {rval:.3f}; difference {difference:+.3f})."
            )

    timestamp = html.escape(str(data.get("timestamp", "Not provided")))
    config = data.get("config", {})
    config_html = "<ul>" + "".join(
        f"<li><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</li>"
        for k, v in config.items()
    ) + "</ul>" if config else "<p>No configuration metadata was provided.</p>"

    chart_html = "".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{html.escape(filename)}" alt="{html.escape(title)}"></section>'
        for title, filename in chart_files
    )
    table_html = (
        "<table><thead><tr><th>Metric</th><th>Fine-tuned</th><th>RAG</th><th>Higher score</th></tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in table_rows
        )
        + "</tbody></table>"
    )
    warning_html = (
        "<section class='warning'><h2>Data-quality notes</h2><ul>"
        + "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
        + "</ul></section>"
        if warnings else ""
    )
    observations_html = "<ul>" + "".join(f"<li>{html.escape(o.replace('**', ''))}</li>" for o in observations) + "</ul>"

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fine-tuned vs RAG Benchmark Report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 32px auto; padding: 0 20px; line-height: 1.55; color: #202124; }}
h1, h2 {{ line-height: 1.2; }}
.meta {{ color: #555; }}
section {{ margin: 30px 0; }}
img {{ display: block; width: 100%; max-width: 1100px; height: auto; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 9px; text-align: left; }}
th {{ background: #f2f2f2; }}
.warning {{ padding: 12px 18px; border: 1px solid #b8860b; background: #fff8e1; }}
small {{ color: #555; }}
</style>
</head>
<body>
<h1>Fine-tuned vs RAG Benchmark</h1>
<p class="meta">Evaluation timestamp: {timestamp} · Aggregate samples reported: {html.escape(str(declared_n if declared_n is not None else "unknown"))}</p>
{warning_html}
<section><h2>Aggregate metric scores</h2>{table_html}</section>
<section><h2>Automatic observations</h2>{observations_html}</section>
{chart_html}
<section><h2>Run configuration</h2>{config_html}</section>
<section><h2>Interpretation and limitations</h2>
<ul>
<li>Faithfulness estimates whether answer claims are supported by the contexts supplied to the evaluator; it is not a general correctness score.</li>
<li>Answer relevancy estimates whether an answer addresses the question; it does not establish medical or factual correctness.</li>
<li>Context precision applies to retrieval and is not directly comparable to a fine-tuned answer score when the fine-tuned system does not retrieve contexts.</li>
<li>Latency from the CSV is based on the recorded per-row values. Confirm both systems were timed under comparable conditions.</li>
<li>Small sample sizes are exploratory. Do not claim one approach is generally superior from only a few examples.</li>
<li>Inspect the evaluation code to confirm the fine-tuned model is scored against meaningful evidence contexts. Using the reference answer as a substitute context can distort faithfulness results.</li>
</ul></section>
<footer><small>Generated automatically by visualize_benchmark.py</small></footer>
</body></html>"""
    report_path = out / "benchmark_report.html"
    report_path.write_text(report, encoding="utf-8")

    # Draft social post. Keep it conservative and do not overclaim from tiny samples.
    post_lines = [
        "# Fine-tuned LLM vs RAG — benchmark snapshot",
        "",
        "I’m benchmarking a domain fine-tuned model against a Retrieval-Augmented Generation (RAG) pipeline.",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Fine-tuned | RAG |",
        "|---|---:|---:|",
    ]
    for metric in all_metrics:
        post_lines.append(f"| {metric.replace('_', ' ').title()} | {fmt(ft.get(metric))} | {fmt(rag.get(metric))} |")
    post_lines += [
        "",
        "## What I’m measuring",
        "- Faithfulness: support from the provided evidence contexts.",
        "- Answer relevancy: how directly the response addresses the question.",
        "- Context precision: relevance of retrieved context, for the RAG pipeline.",
        "- Latency: recorded response-generation time.",
        "",
        "## Notes",
        f"- Evaluation timestamp: {data.get('timestamp', 'not recorded')}",
        f"- Samples reported in aggregate JSON: {declared_n if declared_n is not None else 'unknown'}",
        "- These are preliminary results, not a general claim that one architecture is always better.",
        "- I’m checking context alignment, evaluation failures, and sample size before drawing conclusions.",
        "",
        "Visuals and detailed notes are available in `benchmark_report.html`.",
        "",
        "#LLM #RAG #MachineLearning #GenerativeAI #Evaluation",
    ]
    post_path = out / "linkedin_post_draft.md"
    post_path.write_text("\n".join(post_lines) + "\n", encoding="utf-8")

    summary = {
        "input_scores_json": str(scores_path),
        "input_disagreements_csv": str(csv_path),
        "aggregate_samples": declared_n,
        "csv_rows": len(rows),
        "charts": [filename for _, filename in chart_files],
        "report": str(report_path),
        "social_post_draft": str(post_path),
        "warnings": warnings,
    }
    (out / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nVisualization generation complete.")
    print(f"Output directory: {out}")
    for title, filename in chart_files:
        print(f"  Chart:  {out / filename}")
    print(f"  Report: {report_path}")
    print(f"  Post:   {post_path}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
