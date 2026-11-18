"""
utils/rag_utils.py — RAG-specific RAGAS dataset builder.

The build_ragas_dataset_rag function generates answers via the RAG
pipeline and packages them into a RAGAS-compatible dataset.

The run_ragas function is now a thin wrapper that delegates to
evaluation.run_ragas._run_ragas — the judge factory and config
live in evaluation/judge.py and evaluation/config.py.

This file is kept for backward compatibility with existing imports.
"""

from pathlib import Path
from rich.console import Console
from rich.progress import track
import warnings

console = Console()


def build_ragas_dataset_rag(rag_chain, test_samples: list):
    """Generate answers with RAG and build a RAGAS-compatible dataset."""
    from datasets import Dataset

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
