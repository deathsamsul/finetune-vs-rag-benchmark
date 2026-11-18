"""
compare.py — Deep-dive comparison of fine-tuned vs. RAG disagreements.

After running run_ragas.py you have aggregate scores. This script goes
further: it finds the specific questions where the two pipelines diverge
most, prints them side-by-side, and writes a CSV for human review.

Usage:
    python -m evaluation.compare
    python -m evaluation.compare --n_samples 100 --top_n 15
    python -m evaluation.compare --metric faithfulness
    python -m evaluation.compare --judge-provider groq --judge-model llama-3.1-8b-instant
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evaluation.config import EvalConfig
from evaluation.judge import build_judge, build_embeddings
from utils.args import compare_parse_args

load_dotenv()
console = Console()

RESULTS_DIR = Path("evaluation/results")


def load_test_samples(path: str, n: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Test data not found: {p}[/red]")
        console.print("Run: python data/prepare_data.py")
        sys.exit(1)
    try:
        import jsonlines
        with jsonlines.open(p) as r:
            samples = list(r)
    except ImportError:
        with open(p, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
    return samples[:n]


def run_per_sample_ragas(dataset, metric_name: str, judge, judge_embeddings) -> list[float]:
    """
    Run RAGAS on individual samples to get per-sample scores.

    Uses the shared judge instance instead of hardcoding a provider.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    metric_map = {
        "faithfulness":     faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    metric = metric_map[metric_name]
    metric.llm = judge

    if metric_name == "answer_relevancy":
        metric.embeddings = judge_embeddings

    scores = []
    for i in range(len(dataset)):
        single = dataset.select([i])

        try:
            result = evaluate(single, metrics=[metric])
            score = result.to_pandas()[metric_name].iloc[0]
            val = float(score)
            # Don't silently convert NaN to zero
            if val != val:  # NaN check
                console.print(
                    f"[yellow]  Sample {i}: {metric_name} = NaN (judge failed)[/yellow]"
                )
                scores.append(float("nan"))
            else:
                scores.append(val)
        except Exception as e:
            console.print(
                f"[red]Evaluation failed for sample {i}: {e}[/red]"
            )
            scores.append(float("nan"))

    return scores


def generate_answers_finetuned(samples: list[dict], checkpoint_dir: str) -> list[dict]:
    """Run fine-tuned model inference on all samples."""
    try:
        from utils.fine_tune_inference import FineTunedInference
        model = FineTunedInference(checkpoint_dir)
        model.load()
    except FileNotFoundError as e:
        console.print(f"[yellow]Fine-tuned model not found: {e}[/yellow]")
        return [{"answer": "N/A (model not loaded)", "latency_ms": 0}] * len(samples)

    results = []
    for i, s in enumerate(samples):
        console.print(f"  FT inference {i+1}/{len(samples)}...", end="\r")
        try:
            r = model.generate(s["question"])
            results.append(r)
        except Exception as e:
            results.append({"answer": f"ERROR: {e}", "latency_ms": 0})
    console.print()
    return results


def generate_answers_rag(samples: list[dict], index_path: str) -> list[dict]:
    """Run RAG pipeline inference on all samples."""
    try:
        from utils.rag.rag_chain import RAGChain
        chain = RAGChain(index_path=index_path)
        chain.load()
    except FileNotFoundError as e:
        console.print(f"[yellow]RAG index not found: {e}[/yellow]")
        return [{"answer": "N/A (index not built)", "latency_ms": 0,
                 "contexts": [""]}] * len(samples)

    results = []
    for i, s in enumerate(samples):
        console.print(f"  RAG inference {i+1}/{len(samples)}...", end="\r")
        try:
            r = chain.query(s["question"])
            results.append(r)
        except Exception as e:
            results.append({"answer": f"ERROR: {e}", "latency_ms": 0,
                            "contexts": [""]})
    console.print()
    return results


def _score_both_metrics(samples, ft_results, rag_results,
                        judge, judge_embeddings) -> list[dict]:
    """Score both pipelines on faithfulness and answer relevancy."""
    from datasets import Dataset
    import math

    metric_names = ["faithfulness", "answer_relevancy"]
    all_scores = {}

    for metric_name in metric_names:
        console.print(f"\n[bold]Evaluating {metric_name}...[/bold]")

        ft_ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": ft["answer"],
                "contexts": ft.get("contexts", []),
                "ground_truth": s["answer"],
            }
            for s, ft in zip(samples, ft_results)
        ])

        rag_ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": rag["answer"],
                "contexts": rag.get("contexts", []),
                "ground_truth": s["answer"],
            }
            for s, rag in zip(samples, rag_results)
        ])

        all_scores[f"ft_{metric_name}"] = run_per_sample_ragas(
            ft_ds, metric_name, judge, judge_embeddings
        )
        all_scores[f"rag_{metric_name}"] = run_per_sample_ragas(
            rag_ds, metric_name, judge, judge_embeddings
        )

    rows = []
    for i, (s, ft, rag) in enumerate(zip(samples, ft_results, rag_results)):
        row = {
            "question": s["question"],
            "ground_truth": s["answer"],
            "ft_answer": ft["answer"],
            "rag_answer": rag["answer"],
            "ft_latency_ms": ft.get("latency_ms", 0),
            "rag_latency_ms": rag.get("latency_ms", 0),
            "rag_sources": "|".join(rag.get("sources", [])),
        }

        for metric_name in metric_names:
            ft_score = all_scores[f"ft_{metric_name}"][i]
            rag_score = all_scores[f"rag_{metric_name}"][i]

            row[f"ft_{metric_name}"] = ft_score
            row[f"rag_{metric_name}"] = rag_score
            # Only compute delta if both scores are valid
            if math.isnan(ft_score) or math.isnan(rag_score):
                row[f"{metric_name}_delta"] = float("nan")
            else:
                row[f"{metric_name}_delta"] = ft_score - rag_score

        rows.append(row)

    # Sort by largest valid difference
    def _sort_key(r):
        deltas = []
        for mn in metric_names:
            d = r.get(f"{mn}_delta", 0)
            if isinstance(d, float) and d == d:  # not NaN
                deltas.append(abs(d))
        return max(deltas) if deltas else 0

    rows.sort(key=_sort_key, reverse=True)
    return rows


def print_disagreements(rows: list[dict], top_n: int, metric_name: str):
    """Print the top-N disagreements in a rich table."""
    console.print(f"\n[bold]Top {top_n} disagreements by |Δ{metric_name}|:[/bold]\n")

    for rank, row in enumerate(rows[:top_n], start=1):
        delta = row.get("delta", row.get(f"{metric_name}_delta", 0))
        ft_score = row.get("ft_score", row.get(f"ft_{metric_name}", 0))
        rag_score = row.get("rag_score", row.get(f"rag_{metric_name}", 0))

        winner = "🏆 Fine-Tuned" if delta > 0 else "🏆 RAG"
        panel_content = (
            f"[bold]Q:[/bold] {row['question'][:120]}...\n\n"
            f"[bold cyan]Fine-Tuned[/bold cyan] ({metric_name}: {ft_score:.3f}):\n"
            f"  {row['ft_answer'][:200]}...\n\n"
            f"[bold green]RAG[/bold green] ({metric_name}: {rag_score:.3f}):\n"
            f"  {row['rag_answer'][:200]}...\n\n"
            f"[dim]Δ = {delta:+.3f}  →  {winner}[/dim]"
        )
        console.print(Panel(
            panel_content,
            title=f"[bold]#{rank}[/bold]  |Δ| = {abs(delta):.3f}",
            border_style="yellow" if abs(delta) > 0.3 else "dim",
            padding=(0, 1),
        ))


def print_aggregate_stats(rows: list[dict], metric_name: str):
    """Print aggregate win/loss stats across all compared samples."""
    import math

    # Get the right keys
    ft_key = "ft_score" if "ft_score" in rows[0] else f"ft_{metric_name}"
    rag_key = "rag_score" if "rag_score" in rows[0] else f"rag_{metric_name}"
    delta_key = "delta" if "delta" in rows[0] else f"{metric_name}_delta"

    # Filter to valid rows only
    valid_rows = [
        r for r in rows
        if not math.isnan(r.get(ft_key, float("nan")))
        and not math.isnan(r.get(rag_key, float("nan")))
    ]

    if not valid_rows:
        console.print("[yellow]No valid scores to aggregate.[/yellow]")
        return

    ft_wins   = sum(1 for r in valid_rows if r[delta_key] > 0.05)
    rag_wins  = sum(1 for r in valid_rows if r[delta_key] < -0.05)
    ties      = len(valid_rows) - ft_wins - rag_wins

    avg_ft    = sum(r[ft_key]  for r in valid_rows) / len(valid_rows)
    avg_rag   = sum(r[rag_key] for r in valid_rows) / len(valid_rows)
    avg_ft_ms = sum(r["ft_latency_ms"]  for r in valid_rows) / len(valid_rows)
    avg_rag_ms= sum(r["rag_latency_ms"] for r in valid_rows) / len(valid_rows)

    table = Table(
        title=f"Aggregate Comparison — {metric_name} (n={len(valid_rows)})",
        header_style="bold cyan",
    )
    table.add_column("",            style="bold", min_width=22)
    table.add_column("Fine-Tuned",  justify="center", min_width=14)
    table.add_column("RAG",         justify="center", min_width=14)
    table.add_column("Better",      justify="center", min_width=12)

    overall_winner_score = "Fine-Tuned" if avg_ft > avg_rag else "RAG"
    overall_winner_lat   = "Fine-Tuned" if avg_ft_ms < avg_rag_ms else "RAG"

    table.add_row(
        f"Avg {metric_name}",
        f"{avg_ft:.3f}", f"{avg_rag:.3f}",
        f"🏆 {overall_winner_score}",
    )
    table.add_row(
        "Avg latency (ms)",
        f"{avg_ft_ms:.0f}", f"{avg_rag_ms:.0f}",
        f"🏆 {overall_winner_lat}",
    )
    table.add_row(
        "Wins (Δ > 0.05)",
        str(ft_wins), str(rag_wins),
        f"{'🏆 FT' if ft_wins > rag_wins else '🏆 RAG'} ({ties} ties)",
    )

    invalid_count = len(rows) - len(valid_rows)
    if invalid_count > 0:
        table.add_row(
            "Invalid (NaN)", str(invalid_count), "", "⚠ judge failures"
        )

    console.print(table)


def write_csv(rows: list[dict], output_path: str):
    """Write all comparison rows to CSV for manual review."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        console.print("[yellow]No rows to write to CSV.[/yellow]")
        return

    fieldnames = ["rank"] + [k for k in rows[0].keys()]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    console.print(f"\n  [green]✓ CSV saved: {p}[/green]")
    console.print("  Open in Excel/Sheets to review disagreements.")


def main():
    args = compare_parse_args()
    cfg = EvalConfig.from_env_and_args(args)

    console.print(Panel.fit(
        "[bold blue]🔍 Pipeline Disagreement Analysis[/bold blue]\n"
        f"[dim]Judge: {cfg.summary_str()}  |  "
        f"Finding where fine-tuned and RAG diverge most[/dim]"
    ))

    # Load test data
    console.print(f"\n[bold]Step 1/4 — Loading {args.n_samples} test samples...[/bold]")
    samples = load_test_samples(args.test_data, args.n_samples)
    console.print(f"  Loaded {len(samples)} samples")

    # Generate answers
    console.print(f"\n[bold]Step 2/4 — Fine-tuned inference...[/bold]")
    ft_results = generate_answers_finetuned(samples, args.finetuned_model)

    console.print(f"\n[bold]Step 3/4 — RAG inference...[/bold]")
    rag_results = generate_answers_rag(samples, args.rag_index)

    # Build judge ONCE and reuse
    console.print(f"\n[bold]Step 4/4 — Scoring with RAGAS ({args.metric})...[/bold]")
    judge = build_judge(cfg)
    judge_embeddings = build_embeddings(cfg)

    rows = _score_both_metrics(
        samples, ft_results, rag_results,
        judge, judge_embeddings,
    )

    # Print results
    console.print("\n")
    print_aggregate_stats(rows, args.metric)
    print_disagreements(rows, args.top_n, args.metric)
    write_csv(rows, args.output_csv)

    console.print(Panel.fit(
        "[bold green]✓ Comparison complete![/bold green]\n\n"
        f"CSV for human review: [bold]{args.output_csv}[/bold]\n\n"
        "Key takeaways to include in your LinkedIn post:\n"
        "  • Which pipeline won on faithfulness?\n"
        "  • Which pipeline won on latency?\n"
        "  • What types of questions each approach struggles with?"
    ))


if __name__ == "__main__":
    main()
