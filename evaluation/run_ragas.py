"""
run_ragas.py — RAGAS evaluation for both fine-tuned and RAG pipelines.

Runs both pipelines on the held-out test set and evaluates
with RAGAS faithfulness, answer_relevancy, and context_precision.

RAGAS metrics:
    faithfulness:       fraction of claims in the answer supported by context
    answer_relevancy:   how directly the answer addresses the question
    context_precision:  whether relevant retrieved chunks are ranked highest

Usage:
    python evaluation/run_ragas.py
    python evaluation/run_ragas.py --n_samples 50   # quick test on 50 samples
    python evaluation/run_ragas.py --skip_finetuned  # RAG only

Outputs:
    evaluation/results/ragas_scores.json
    evaluation/results/benchmark_report.md
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

load_dotenv()
console = Console()

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True) # parents=true if parent directories don't exist, create them , exist_ok= ture if alredy exist then not raise error 


# argparse — Python's built-in module for reading arguments provided when you run a script from a terminal.
def parse_args():  # It creates a command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on fine-tuned and RAG pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    
    parser.add_argument("--finetuned_model", type=str,
                        default="./checkpoints/medquad-lora",
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--rag_index", type=str,
                        default="./indexes/medquad_faiss",
                        help="Path to FAISS index directory")
    parser.add_argument("--test_data", type=str,
                        default="data/processed/test.jsonl",
                        help="Path to held-out test JSONL")
    parser.add_argument("--metrics", nargs="+",
                        default=["faithfulness", "answer_relevancy", "context_precision"],
                        help="RAGAS metrics to compute")
    parser.add_argument("--output", type=str,
                        default="evaluation/results/ragas_scores.json",
                        help="Output JSON path")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Evaluate on first N samples (None = all)")
    parser.add_argument("--skip_finetuned", action="store_true",
                        help="Skip fine-tuned pipeline (RAG only)")
    parser.add_argument("--skip_rag", action="store_true",
                        help="Skip RAG pipeline (fine-tuned only)")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of chunks to retrieve for RAG")
    return parser.parse_args()


# Fine-Tuned Inference ---------------------------------------------------
class FineTunedInference:
    """Wraps the QLoRA fine-tuned model for batch inference."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model = None
        self._tokenizer = None

    def load(self):
        """Load the fine-tuned model + adapter."""
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_dir}\n"
                "Run: python finetuning/train_lora.py  (or make finetune)" )
        

        console.print(f"  Loading fine-tuned model from {self.checkpoint_dir}...")

        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.checkpoint_dir),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True, )
            
            FastLanguageModel.for_inference(model)

        except ImportError:
            # Fallback to standard HuggingFace PEFT loading
            console.print("  [yellow]Unsloth not found; using vanilla PEFT loader[/yellow]")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            base_cfg_path = self.checkpoint_dir / "adapter_config.json"
            with open(base_cfg_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg["base_model_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
            )
            model = PeftModel.from_pretrained(base, str(self.checkpoint_dir))
            model.eval()

        self._model = model
        self._tokenizer = tokenizer
        console.print("  [green]✓ Fine-tuned model loaded[/green]")
        return self

    def generate(self, question: str) -> dict:
        """Generate an answer for a single question."""
        from finetuning.utils import format_for_inference
        import torch

        prompt = format_for_inference(question)
        inputs = self._tokenizer(prompt, return_tensors="pt")

        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the new tokens (not the prompt)
        n_input_tokens = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][n_input_tokens:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "answer":     answer,
            "latency_ms": latency_ms,
            "n_tokens":   len(new_tokens), }



# RAGAS Evaluation Helpers------------------------------------------------------
def build_ragas_dataset_rag(
    rag_chain, test_samples: list[dict], ) -> "datasets.Dataset":
    """
    Run the RAG pipeline on the test set and format results for RAGAS.
    RAGAS expects a Dataset with columns:
        question      str    The input question
        answer        str    The generated answer
        contexts      list   Retrieved chunks used to generate the answer
        ground_truth  str    The reference answer (from test set)
    """
    from datasets import Dataset

    rows = []
    for sample in track(test_samples, description="  RAG inference..."):
        try:
            result = rag_chain.query(sample["question"])
            rows.append({
                "question":    sample["question"],
                "answer":      result["answer"],
                "contexts":    result["contexts"],
                "ground_truth": sample["answer"],
            })
        except Exception as e:
            console.print(f"  [yellow]Error on sample: {e}[/yellow]")
            # Don't silently drop — include with empty answer so RAGAS scores it low
            rows.append({
                "question":    sample["question"],
                "answer":      "",
                "contexts":    [""],
                "ground_truth": sample["answer"],
            })

    return Dataset.from_list(rows)


def build_ragas_dataset_finetuned(
    ft_model: FineTunedInference,
    test_samples: list[dict],) -> "datasets.Dataset":
    """
    Run fine-tuned model inference and format for RAGAS.
    Note: Fine-tuned model has no retrieval step, so 'contexts' is set
    to the training answer (closest we have to a reference context).
    Context precision is reported as N/A for this pipeline.
    """
    from datasets import Dataset

    rows = []
    for sample in track(test_samples, description="  Fine-tuned inference..."):
        try:
            result = ft_model.generate(sample["question"])
            rows.append({
                "question":    sample["question"],
                "answer":      result["answer"],
                # No retrieval step — context is the reference answer itself.
                # Faithfulness for fine-tuned model measures internal consistency.
                "contexts":    [sample["answer"]],
                "ground_truth": sample["answer"],
            })
        except Exception as e:
            console.print(f"  [yellow]Error on sample: {e}[/yellow]")
            rows.append({
                "question":    sample["question"],
                "answer":      "",
                "contexts":    [""],
                "ground_truth": sample["answer"],
            })

    return Dataset.from_list(rows)

# --------- rugs for openai evaluation--------------
# def run_ragas(
#     dataset,
#     metrics: list[str],
#     pipeline_name: str, ) -> dict:
#     """
#     Run RAGAS evaluation on a dataset.
#     Returns a dict mapping metric names to scores.
#     """
#     from ragas import evaluate
#     from ragas.metrics import (
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#     )

#     metric_map = {
#         "faithfulness":      faithfulness,
#         "answer_relevancy":  answer_relevancy,
#         "context_precision": context_precision,
#     }

#     selected = [metric_map[m] for m in metrics if m in metric_map]
#     # context_precision makes no sense for fine-tuned (no real retrieval step)
#     if pipeline_name == "finetuned" and context_precision in selected:
#         selected.remove(context_precision)
#         console.print("  [dim]context_precision skipped for fine-tuned (no retrieval)[/dim]")

#     console.print(f"  Running RAGAS with {len(selected)} metrics...")
#     result = evaluate(dataset, metrics=selected)

#     # Convert to plain dict with float values
#     scores = {}
#     for metric in selected:
#         key = metric.name
#         scores[key] = float(result[key])

#     return scores
# ----- google gemini evaluation-----------
def run_ragas(
    dataset,
    metrics: list[str],
    pipeline_name: str,
) -> dict:
    """Run RAGAS evaluation using Gemini and return mean metric scores."""
    import os
    from dotenv import load_dotenv
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    )

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }

    selected = [
        metric_map[m] for m in metrics if m in metric_map
    ]

    if pipeline_name == "finetuned" and context_precision in selected:
        selected.remove(context_precision)
        console.print(
            "  [dim]context_precision skipped for fine-tuned "
            "(no retrieval)[/dim]"
        )

    console.print(
        f"  Running RAGAS with {len(selected)} metrics using Gemini..."
    )

    result = evaluate(
        dataset,
        metrics=selected,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    scores = {}
    for metric in selected:
        key = metric.name
        values = [
            float(value)
            for value in result[key]
            if value is not None
        ]
        scores[key] = (
            sum(values) / len(values) if values else None
        )

    return scores


def collect_latency_stats(samples_queried: list[dict]) -> dict:
    """Aggregate latency statistics from per-sample results."""
    latencies = [s["latency_ms"] for s in samples_queried if "latency_ms" in s]
    if not latencies:
        return {}
    return {
        "avg_latency_ms":    round(sum(latencies) / len(latencies), 1),
        "p50_latency_ms":    round(sorted(latencies)[len(latencies) // 2], 1),
        "p95_latency_ms":    round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        "max_latency_ms":    round(max(latencies), 1),
    }


# Reporting----------------------------------------------------------------------------------------
def print_results_table(results: dict):
    """Print a rich comparison table to the terminal."""
    table = Table(
        title=" RAGAS Benchmark Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold", min_width=22)
    table.add_column("Fine-Tuned (QLoRA)", justify="center", min_width=18)
    table.add_column("RAG (FAISS)", justify="center", min_width=14)
    table.add_column("Winner", justify="center", min_width=10)

    ft_scores  = results.get("finetuned", {})
    rag_scores = results.get("rag", {})

    all_metrics = sorted(
        set(ft_scores.keys()) | set(rag_scores.keys())
    )

    for metric in all_metrics:
        ft_val  = ft_scores.get(metric)
        rag_val = rag_scores.get(metric)

        ft_str  = f"{ft_val:.3f}"  if ft_val is not None else "N/A"
        rag_str = f"{rag_val:.3f}" if rag_val is not None else "N/A"

        if ft_val is not None and rag_val is not None:
            winner = "🏆 FT" if ft_val > rag_val else "🏆 RAG"
        else:
            winner = "—"

        table.add_row(metric, ft_str, rag_str, winner)

    console.print(table)


def generate_markdown_report(results: dict, output_path: Path):
    """Auto-generate a markdown benchmark report."""
    ft_scores   = results.get("finetuned",   {})
    rag_scores  = results.get("rag",         {})
    ft_latency  = results.get("ft_latency",  {})
    rag_latency = results.get("rag_latency", {})

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# RAGAS Benchmark Report",
        f"",
        f"Generated: {now}",
        f"",
        "## Metric Scores",
        "",
        "| Metric | Fine-Tuned (QLoRA) | RAG (FAISS + LangChain) | Winner |",
        "|---|---|---|---|",
    ]

    all_metrics = sorted(set(ft_scores.keys()) | set(rag_scores.keys()))
    for metric in all_metrics:
        ft_val  = ft_scores.get(metric)
        rag_val = rag_scores.get(metric)
        ft_str  = f"{ft_val:.3f}" if ft_val is not None else "N/A"
        rag_str = f"{rag_val:.3f}" if rag_val is not None else "N/A"
        if ft_val is not None and rag_val is not None:
            winner = "Fine-Tuned" if ft_val > rag_val else "RAG"
        else:
            winner = "—"
        lines.append(f"| {metric} | {ft_str} | {rag_str} | {winner} |")

    lines += [
        "",
        "## Latency",
        "",
        "| Metric | Fine-Tuned | RAG |",
        "|---|---|---|",
    ]

    lat_metrics = sorted(set(ft_latency.keys()) | set(rag_latency.keys()))
    for m in lat_metrics:
        ft_val  = ft_latency.get(m,  "—")
        rag_val = rag_latency.get(m, "—")
        lines.append(f"| {m} | {ft_val} | {rag_val} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Faithfulness** measures hallucination — higher = fewer hallucinated claims.",
        "- **Answer Relevancy** measures on-topic responses — higher = more direct answers.",
        "- **Context Precision** (RAG only) measures retrieval quality — higher = relevant chunks ranked first.",
        "",
        "## Notes",
        "",
        "- Test set is held out from fine-tuning — no data leakage between pipelines.",
        "- RAGAS uses an LLM judge internally — treat scores as directional (±3-5%).",
        "- Context precision is N/A for the fine-tuned model (no retrieval step).",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"  [green]✓ Report saved: {output_path}[/green]")



# Main--------------------------------------------------------------------------------
def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold blue]🔬 RAGAS Evaluation — Fine-Tuned vs. RAG[/bold blue]\n"
        "[dim]Day 2: Head-to-head benchmark[/dim]"
    ))

    #  Load test set -------------------------------------------------------
    import jsonlines
    test_path = Path(args.test_data)
    if not test_path.exists():
        console.print(f"[red]Test data not found: {test_path}[/red]")
        console.print("Run: python data/prepare_data.py")
        sys.exit(1)

    with jsonlines.open(test_path) as reader:
        test_samples = list(reader)

    if args.n_samples:
        test_samples = test_samples[:args.n_samples]
        console.print(f"  Using [bold]{args.n_samples}[/bold] samples (--n_samples flag)")

    console.print(f"  Test samples: [bold]{len(test_samples):,}[/bold]")

    results = {}
    raw_results = {}

    #  Fine-Tuned Pipeline ---------------------------
    if not args.skip_finetuned:
        console.print(f"\n[bold]━━ Fine-Tuned Pipeline (QLoRA) ━━[/bold]")
        try:
            ft_model = FineTunedInference(args.finetuned_model)
            ft_model.load()

            ft_dataset = build_ragas_dataset_finetuned(ft_model, test_samples)
            ft_scores  = run_ragas(ft_dataset, args.metrics, "finetuned")

            # Latency from raw inference (stored in ft_dataset rows)
            results["finetuned"] = ft_scores
            raw_results["finetuned_dataset_size"] = len(test_samples)
            console.print(f"  Fine-tuned scores: {ft_scores}")

        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping fine-tuned: {e}[/yellow]")
            results["finetuned"] = {}

    #  RAG Pipeline -------------------------
    if not args.skip_rag:
        console.print(f"\n[bold]━━ RAG Pipeline (FAISS + LangChain) ━━[/bold]")
        try:
            from rag.rag_chain import RAGChain
            rag_chain = RAGChain(index_path=args.rag_index, k=args.k)
            rag_chain.load()

            rag_dataset = build_ragas_dataset_rag(rag_chain, test_samples)
            rag_scores  = run_ragas(rag_dataset, args.metrics, "rag")

            results["rag"] = rag_scores
            console.print(f"  RAG scores: {rag_scores}")

        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping RAG: {e}[/yellow]")
            results["rag"] = {}

    #  Print comparison ---------------------------------
    console.print("\n")
    print_results_table(results)

    #  Save JSON ---------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_output = {
        "timestamp":  datetime.now().isoformat(),
        "n_samples":  len(test_samples),
        "scores":     results,
        "config": {
            "finetuned_model": args.finetuned_model,
            "rag_index":       args.rag_index,
            "metrics":         args.metrics,
            "k":               args.k,
        }
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
    console.print(f"\n  [green]✓ Scores saved: {output_path}[/green]")

    #  Generate markdown report -----------------------------------------
    report_path = RESULTS_DIR / "benchmark_report.md"
    generate_markdown_report(results, report_path)

    console.print(
        Panel.fit(
            "[bold green]✓ Evaluation complete![/bold green]\n\n"
            f"Results: [bold]{output_path}[/bold]\n"
            f"Report:  [bold]{report_path}[/bold]\n\n"
            "Next step:\n"
            "  [bold]make serve[/bold]   — start FastAPI + Langfuse tracing"
        )
    )


if __name__ == "__main__":
    main()
