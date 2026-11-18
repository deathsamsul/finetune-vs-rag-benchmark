"""
run_ragas.py — RAGAS evaluation for both fine-tuned and RAG pipelines.

Supports dynamic judge selection via CLI or .env:

    # Default local judge (no API key needed)
    python -m evaluation.run_ragas

    # Local 3B judge
    python -m evaluation.run_ragas --judge-provider local \
        --judge-model unsloth/Llama-3.2-3B-Instruct-bnb-4bit

    # Groq cloud judge
    python -m evaluation.run_ragas --judge-provider groq \
        --judge-model llama-3.1-8b-instant

    # OpenRouter judge
    python -m evaluation.run_ragas --judge-provider openrouter \
        --judge-model openai/gpt-4o-mini

    # Quick test
    python -m evaluation.run_ragas --n_samples 2

Outputs:
    evaluation/results/ragas_scores.json
    evaluation/results/benchmark_report.md
"""

import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from evaluation.config import EvalConfig
from evaluation.judge import build_judge, build_embeddings
from utils.args import parse_args
from utils.fine_tune_inference import FineTunedInference
from utils.report_utils import generate_markdown_report, print_results_table

load_dotenv()

console = Console()
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# RAGAS metrics
# ---------------------------------------------------------------------------

def _get_metrics():
    """Import and return RAGAS metric objects."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    return faithfulness, answer_relevancy, context_precision


def _extract_score(result, name: str) -> float:
    """Extract float score from RAGAS result (handles list or scalar)."""
    val = result[name]
    if isinstance(val, list):
        valid = [float(v) for v in val
                 if v is not None and str(v) not in ("nan", "NaN", "None")]
        return sum(valid) / len(valid) if valid else float("nan")
    return float(val) if val is not None else float("nan")


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_ragas_dataset_finetuned(ft_model, test_samples: list):
    """Generate answers with the fine-tuned model and build a RAGAS dataset."""
    from datasets import Dataset
    from rich.progress import track

    rows = []
    for sample in track(test_samples, description="  Fine-tuned inference..."):
        try:
            r = ft_model.generate(sample["question"])
            rows.append({
                "question":     sample["question"],
                "answer":       r["answer"],
                "contexts":     [sample["answer"]],
                "ground_truth": sample["answer"],
            })
        except Exception as e:
            console.print(f"  [yellow]FT error: {e}[/yellow]")
            rows.append({
                "question": sample["question"], "answer": "",
                "contexts": [""], "ground_truth": sample["answer"],
            })
    return Dataset.from_list(rows)


def _build_ragas_dataset_rag(rag_chain, test_samples: list):
    """Generate answers with the RAG pipeline and build a RAGAS dataset."""
    from datasets import Dataset
    from rich.progress import track

    rows = []
    for sample in track(test_samples, description="  RAG inference..."):
        try:
            r = rag_chain.query(sample["question"])
            rows.append({
                "question":     sample["question"],
                "answer":       r["answer"],
                "contexts":     r["contexts"] or [""],
                "ground_truth": sample["answer"],
            })
        except Exception as e:
            console.print(f"  [yellow]RAG error: {e}[/yellow]")
            rows.append({
                "question": sample["question"], "answer": "",
                "contexts": [""], "ground_truth": sample["answer"],
            })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------

def _run_ragas(dataset, metrics: list, pipeline_name: str,
               cfg: EvalConfig, judge, embeddings) -> dict:
    """
    Run RAGAS evaluation with the configured judge and local embeddings.
    Returns {metric_name: float_score}.
    """
    faithfulness, answer_relevancy, context_precision = _get_metrics()

    metric_map = {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
    }
    selected = [metric_map[m] for m in metrics if m in metric_map]

    # context_precision doesn't apply to fine-tuned (no retrieval step)
    if pipeline_name == "finetuned" and context_precision in selected:
        selected.remove(context_precision)
        console.print("  [dim]context_precision skipped for fine-tuned "
                       "(no retrieval step)[/dim]")

    console.print(
        f"  Running RAGAS ({len(selected)} metrics)  "
        f"[dim]judge: {cfg.summary_str()}  |  embeddings: {cfg.embed_model}[/dim]"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas import evaluate

        result = evaluate(
            dataset,
            metrics=selected,
            llm=judge,
            embeddings=embeddings,
            raise_exceptions=False,
        )

    scores = {}
    has_failure = False
    for metric in selected:
        key = metric.name
        score = _extract_score(result, key)

        if math.isnan(score):
            has_failure = True
            console.print(
                f"  [yellow]⚠  {key} = NaN — judge ({cfg.summary_str()}) "
                f"failed to score this metric.[/yellow]"
            )
            # Store NaN, NOT zero — caller must handle it
            scores[key] = float("nan")
        else:
            console.print(f"  [green]✓[/green]  {key} = [bold]{score:.3f}[/bold]")
            scores[key] = score

    if has_failure:
        console.print(
            "  [yellow]Some metrics returned NaN. Check the judge provider "
            "and API key/quota.[/yellow]"
        )

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = EvalConfig.from_env_and_args(args)

    # Validate config
    errors = cfg.validate()
    if errors:
        for e in errors:
            console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold blue]🔬 RAGAS Evaluation — Fine-Tuned vs. RAG[/bold blue]\n"
        f"[dim]Judge: {cfg.summary_str()}  |  "
        f"Embeddings: {cfg.embed_model} (local)[/dim]"
    ))

    # Load test data
    test_path = Path(cfg.test_data)
    if not test_path.exists():
        console.print(f"[red]Test data not found: {test_path}[/red]")
        sys.exit(1)

    try:
        import jsonlines
        with jsonlines.open(test_path) as r:
            test_samples = list(r)
    except ImportError:
        with open(test_path, "r", encoding="utf-8") as f:
            test_samples = [json.loads(line) for line in f if line.strip()]

    if cfg.n_samples:
        test_samples = test_samples[:cfg.n_samples]
        console.print(f"  Using [bold]{cfg.n_samples}[/bold] samples")
    console.print(f"  Test samples: [bold]{len(test_samples):,}[/bold]")

    # Build judge and embeddings ONCE — reused across both pipelines
    judge = build_judge(cfg)
    embeddings = build_embeddings(cfg)

    results = {}

    # --- Fine-Tuned pipeline ---
    if not cfg.skip_finetuned:
        console.print("\n[bold]━━ Fine-Tuned Pipeline (QLoRA) ━━[/bold]")
        try:
            ft = FineTunedInference(cfg.finetuned_model)
            ft.load()
            ds = _build_ragas_dataset_finetuned(ft, test_samples)
            results["finetuned"] = _run_ragas(
                ds, cfg.metrics, "finetuned", cfg, judge, embeddings
            )
        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping: {e}[/yellow]")
            results["finetuned"] = {}

        # Release VRAM before RAG pipeline
        import gc
        import torch
        if "ft" in locals():
            del ft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- RAG pipeline ---
    if not cfg.skip_rag:
        console.print("\n[bold]━━ RAG Pipeline (FAISS + LangChain) ━━[/bold]")
        try:
            from utils.rag.rag_chain import RAGChain
            rag_chain = RAGChain(
                index_path=cfg.rag_index,
                model_name=str(Path(cfg.finetuned_model).resolve()),
                k=cfg.k,
            )
            rag_chain.load()
            ds = _build_ragas_dataset_rag(rag_chain, test_samples)
            results["rag"] = _run_ragas(
                ds, cfg.metrics, "rag", cfg, judge, embeddings
            )
        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping: {e}[/yellow]")
            results["rag"] = {}

    # --- Save results ---
    console.print()
    print_results_table(results)

    # Check if we have enough valid results to declare winners
    _check_result_validity(results)

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp":  datetime.now().isoformat(),
            "n_samples":  len(test_samples),
            "scores":     results,
            "config": {
                "finetuned_model": cfg.finetuned_model,
                "rag_index":       cfg.rag_index,
                "metrics":         cfg.metrics,
                "k":               cfg.k,
                "judge_provider":  cfg.judge_provider,
                "judge_model":     cfg.judge_model,
                "embeddings":      cfg.embed_model,
            },
        }, f, indent=2)
    console.print(f"\n  [green]✓ Scores saved: {output_path}[/green]")

    generate_markdown_report(results, RESULTS_DIR / "benchmark_report.md", cfg)

    console.print(Panel.fit(
        "[bold green]✓ Evaluation complete![/bold green]\n\n"
        f"Scores : [bold]{output_path}[/bold]\n"
        f"Report : [bold]{RESULTS_DIR / 'benchmark_report.md'}[/bold]"
    ))


def _check_result_validity(results: dict):
    """Warn if results contain NaN — don't declare a winner from invalid data."""
    for pipeline, scores in results.items():
        for metric, value in scores.items():
            if isinstance(value, float) and math.isnan(value):
                console.print(
                    f"  [yellow]⚠ {pipeline}/{metric} is NaN — "
                    f"this result is invalid and should not be compared.[/yellow]"
                )


if __name__ == "__main__":
    main()