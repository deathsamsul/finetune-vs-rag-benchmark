"""
compare.py — Deep-dive comparison of fine-tuned vs. RAG disagreements.

After running run_ragas.py you have aggregate scores. This script goes
further: it finds the specific questions where the two pipelines diverge
most, prints them side-by-side, and writes a CSV for human review.

This is the "interesting" analysis for your LinkedIn post —
showing where each approach wins *and why*.

Usage:
    python evaluation/compare.py
    python evaluation/compare.py --n_samples 100 --top_n 15
    python evaluation/compare.py --metric faithfulness
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
from utils.args import compare_parse_args
from evaluation.scoring import score_answers, both_score_answers



load_dotenv()
console = Console()

RESULTS_DIR = Path("evaluation/results")

# python -m evaluation.compare --n_samples 100 --top_n 15 --metric faithfulness


def load_test_samples(path: str, n: int) -> list[dict]:
    import jsonlines
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Test data not found: {p}[/red]")
        console.print("Run: python data/prepare_data.py")
        sys.exit(1)
    with jsonlines.open(p) as r:
        samples = list(r)
    return samples[:n]


def run_per_sample_ragas(dataset, metric_name: str) -> list[float]:
    """
    Run RAGAS and return a per-sample score list for one metric.
    
    RAGAS by default returns aggregate scores. We call it on single-row
    datasets to get per-sample scores, which is slow but gives us
    the granularity to rank disagreements.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings


    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file.")

    groq_llm = ChatGroq(model=model_name,
                        api_key=api_key,
                        temperature=0,)
    
    evaluator_llm = LangchainLLMWrapper(groq_llm)

    # Use local embeddings instead of OpenAI embeddings.
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
    metric_map = {"faithfulness":     faithfulness,
                   "answer_relevancy": answer_relevancy, }
    
    metric = metric_map[metric_name]
    metric.llm = evaluator_llm       # Explicitly configure the metric's judge and embeddings.

    if metric_name == "answer_relevancy":
        metric.embeddings = evaluator_embeddings

    scores = []
    for i in range(len(dataset)):
        single = dataset.select([i])

        try:
            result = evaluate(single, metrics=[metric])
            score = result.to_pandas()[metric_name].iloc[0]
            scores.append(float(score))
        except Exception as e:
            console.print(
                f"[red]Evaluation failed for sample {i}: {e}[/red]"
            )
            scores.append(0.0)

    return scores




def generate_answers_finetuned(
    samples: list[dict],
    checkpoint_dir: str,
) -> list[dict]:
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


def generate_answers_rag(
    samples: list[dict],
    index_path: str,
) -> list[dict]:
    """Run RAG pipeline inference on all samples."""
    try:
        from utils.rag.rag_chain import RAGChain
        chain = RAGChain(index_path=index_path)
        chain.load()
    except FileNotFoundError as e:
        console.print(f"[yellow]RAG index not found: {e}[/yellow]")
        return [{"answer": "N/A (index not built)", "latency_ms": 0, "contexts": [""]}] * len(samples)

    results = []
    for i, s in enumerate(samples):
        console.print(f"  RAG inference {i+1}/{len(samples)}...", end="\r")
        try:
            r = chain.query(s["question"])
            results.append(r)
        except Exception as e:
            results.append({"answer": f"ERROR: {e}", "latency_ms": 0, "contexts": [""]})
    console.print()
    return results





def print_disagreements(rows: list[dict], top_n: int, metric_name: str):
    """Print the top-N disagreements in a rich table."""
    console.print(f"\n[bold]Top {top_n} disagreements by |Δ{metric_name}|:[/bold]\n")

    for rank, row in enumerate(rows[:top_n], start=1):
        winner = "🏆 Fine-Tuned" if row["delta"] > 0 else "🏆 RAG"
        panel_content = (
            f"[bold]Q:[/bold] {row['question'][:120]}...\n\n"
            f"[bold cyan]Fine-Tuned[/bold cyan] ({metric_name}: {row['ft_score']:.3f}):\n"
            f"  {row['ft_answer'][:200]}...\n\n"
            f"[bold green]RAG[/bold green] ({metric_name}: {row['rag_score']:.3f}):\n"
            f"  {row['rag_answer'][:200]}...\n\n"
            f"[dim]Δ = {row['delta']:+.3f}  →  {winner}[/dim]"
        )
        console.print(Panel(
            panel_content,
            title=f"[bold]#{rank}[/bold]  |Δ| = {abs(row['delta']):.3f}",
            border_style="yellow" if abs(row["delta"]) > 0.3 else "dim",
            padding=(0, 1),
        ))


def print_aggregate_stats(rows: list[dict], metric_name: str):
    """Print aggregate win/loss stats across all compared samples."""
    ft_wins   = sum(1 for r in rows if r["delta"] > 0.05)
    rag_wins  = sum(1 for r in rows if r["delta"] < -0.05)
    ties      = len(rows) - ft_wins - rag_wins

    avg_ft    = sum(r["ft_score"]  for r in rows) / len(rows)
    avg_rag   = sum(r["rag_score"] for r in rows) / len(rows)
    avg_ft_ms = sum(r["ft_latency_ms"]  for r in rows) / len(rows)
    avg_rag_ms= sum(r["rag_latency_ms"] for r in rows) / len(rows)

    table = Table(
        title=f"Aggregate Comparison — {metric_name} (n={len(rows)})",
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
    console.print(table)


def write_csv(rows: list[dict], output_path: str):
    """Write all comparison rows to CSV for manual review."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank", "question", "ground_truth",
        "ft_answer", "rag_answer",
        "ft_score", "rag_score", "delta",
        "ft_latency_ms", "rag_latency_ms", "rag_sources",
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    console.print(f"\n  [green]✓ CSV saved: {p}[/green]")
    console.print(f"  Open in Excel/Sheets to review disagreements.")


def main():
    args = compare_parse_args()

    console.print(Panel.fit(
        "[bold blue]🔍 Pipeline Disagreement Analysis[/bold blue]\n"
        "[dim]Finding where fine-tuned and RAG diverge most[/dim]"
    ))

    # ── Load test data __________-------------------------------------------------
    console.print(f"\n[bold]Step 1/4 — Loading {args.n_samples} test samples...[/bold]")
    samples = load_test_samples(args.test_data, args.n_samples)
    console.print(f"  Loaded {len(samples)} samples")

    # ── Generate answers ----------------------------------------------------------
    console.print(f"\n[bold]Step 2/4 — Fine-tuned inference...[/bold]")
    ft_results = generate_answers_finetuned(samples, args.finetuned_model)

    console.print(f"\n[bold]Step 3/4 — RAG inference...[/bold]")
    rag_results = generate_answers_rag(samples, args.rag_index)

    # ── Score with RAGAS ---------------------------------------------------------------
    console.print(f"\n[bold]Step 4/4 — Scoring with RAGAS ({args.metric})...[/bold]")
    rows = score_answers(samples, ft_results, rag_results,args.metric, console, run_per_sample_ragas)

    # ── Print results ==================================================================
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
