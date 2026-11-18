"""
prepare_data.py — Download MedQuAD and create train/val/test splits.
Downloads the MedQuAD dataset from HuggingFace Hub, cleans it,
and creates instruction-formatted JSONL splits.
Outputs:
    data/processed/train.jsonl   (~80% of data, instruction-formatted)
    data/processed/val.jsonl     (~10% of data)
    data/processed/test.jsonl    (~10% of data, held-out for RAGAS eval)
    data/processed/corpus.jsonl  (all Q&A pairs, used for RAG indexing)
Usage:
    python data/prepare_data.py
    python data/prepare_data.py --max_samples 5000   # faster for testing
"""

import argparse
import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import track 
from utils.args import data_parse_args





load_dotenv()
console = Console()


PROCESSED_DIR = Path("data/processed")
RAW_DIR       = Path("data/raw")


# MedQuAD on HuggingFace:
# https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k
# We use this cleaned medical Q&A dataset as a proxy for MedQuAD
# (MedQuAD itself requires manual download from NIH)
HF_DATASET_NAME = os.getenv('HF_DATASET_NAME',"lavita/ChatDoctor-HealthCareMagic-100k").strip()

# Fallback: if HF_DATASET_NAME unavailable, use this alternative
HF_FALLBACK = os.getenv("HF_FALLBACK","medalpaca/medical_meadow_medqa").strip()


# python data/prepare.py --max_samples 1000 --train_ratio 0.8 --val_ratio 0.1 --seed 42




def download_dataset(dataset_name: str, hf_token: str | None) -> list[dict]:
    """
    Download dataset from HuggingFace and return as list of dicts with keys 'question' and 'answer'.
    """

    from datasets import load_dataset

    console.print(f"  Downloading: [bold]{dataset_name}[/bold]")

    # Pass token only if it's actually set (empty string causes auth errors)
    token = hf_token if hf_token else None

    try:
        ds = load_dataset(dataset_name, token=token, split="train")
    except Exception as e:
        console.print(f"  [yellow]Primary dataset failed ({e}), trying fallback...[/yellow]")
        ds = load_dataset(HF_FALLBACK, token=token, split="train")

    # Normalise column names — different datasets use different names
    samples = []
    for row in track(ds, description="Processing rows..."):
        q = (row.get("input") or row.get("question") or row.get("Question") or row.get("query") or "").strip()
        a = (row.get("output") or row.get("answer") or row.get("Answer") or row.get("response") or "").strip()

        if q and a and len(q) > 10 and len(a) > 20:
            samples.append({"question": q, "answer": a})

    console.print(f"  [green]✓ Loaded {len(samples):,} valid Q&A pairs[/green]")
    return samples



def clean_sample(sample: dict) -> dict | None:
    """
    Clean a single Q&A sample. Returns None to drop the sample.
    Filters:
    - Too short answers (< 30 chars) — likely metadata
    - Too long answers (> 2000 chars) — may cause context overflow
    - Samples with placeholder text
    """

    q = sample["question"].strip()
    a = sample["answer"].strip()

    bad_phrases = [
        "i don't know", "i am not sure", "please consult",
        "i cannot", "as an ai", "i'm sorry, i cannot" ]

    if len(a) < 30 or len(a) > 2000:
        return None
    if any(p in a.lower() for p in bad_phrases):
        return None
    if len(q) < 10:
        return None

    return {"question": q, "answer": a}



def split_dataset(samples: list[dict], train_ratio: float, val_ratio: float, seed: int) -> tuple[list, list, list]:
    """Deterministic train/val/test split."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return train, val, test



def write_jsonl(samples: list[dict], path: Path):
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    console.print(f"  Wrote [bold]{len(samples):,}[/bold] samples → {path}")



def write_corpus(all_samples: list[dict], path: Path):
    """
    Write the full corpus for RAG indexing.
    Each entry has 'id', 'question', 'answer', and 'text' (combined).
    """
    corpus = []
    for i, s in enumerate(all_samples):
        corpus.append({
            "id": f"medqa_{i:06d}",
            "question": s["question"],
            "answer": s["answer"],
            # Combined text for embedding — include both Q and A
            "text": f"Question: {s['question']}\n\nAnswer: {s['answer']}",
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    console.print(f"  Corpus (RAG): [bold]{len(corpus):,}[/bold] documents → {path}")


def print_split_stats(train, val, test):
    """Print a summary of the splits."""
    from rich.table import Table
    table = Table(title="Dataset Splits", show_header=True, header_style="bold cyan")
    table.add_column("Split")
    table.add_column("Samples", justify="right")
    table.add_column("Avg Q length", justify="right")
    table.add_column("Avg A length", justify="right")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        avg_q = sum(len(s["question"]) for s in split) / len(split) if split else 0
        avg_a = sum(len(s["answer"])   for s in split) / len(split) if split else 0
        table.add_row(name, f"{len(split):,}", f"{avg_q:.0f}", f"{avg_a:.0f}")

    console.print(table)




def main():
    args = data_parse_args()

    console.print(Panel.fit(
        "[bold blue] Data Preparation — MedQuAD[/bold blue]\n"
        "[dim]Downloading, cleaning, and splitting the dataset[/dim]" ))


    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if not hf_token:
        console.print("[yellow]Warning: HF_TOKEN not set. Some datasets may be inaccessible.[/yellow]")

    #  Download ----------------------------------------------------------
    console.print("\n[bold]Step 1/4 — Downloading dataset...[/bold]")
    raw_samples = download_dataset(args.dataset, hf_token)

    #  Limit (optional) ----------------------------------------------------------------------------
    if args.max_samples:
        raw_samples = raw_samples[:args.max_samples]
        console.print(f"  [dim]Limited to {args.max_samples:,} samples[/dim]")

    #  Clean -----------------------------------------
    console.print("\n[bold]Step 2/4 — Cleaning...[/bold]")
    cleaned = [c for s in raw_samples if (c := clean_sample(s)) is not None]
    dropped = len(raw_samples) - len(cleaned)
    console.print(f"  Kept: [green]{len(cleaned):,}[/green]  Dropped: [red]{dropped:,}[/red]")

    #  Split ----------------------------------------------------------
    console.print("\n[bold]Step 3/4 — Splitting...[/bold]")
    train, val, test = split_dataset(
        cleaned, args.train_ratio, args.val_ratio, args.seed
    )
    print_split_stats(train, val, test)

    #  Write ----------------------------------------------------------------
    console.print("\n[bold]Step 4/4 — Writing files...[/bold]")
    write_jsonl(train, PROCESSED_DIR / "train.jsonl")
    write_jsonl(val,   PROCESSED_DIR / "val.jsonl")
    write_jsonl(test,  PROCESSED_DIR / "test.jsonl")
    write_corpus(cleaned, PROCESSED_DIR / "corpus.jsonl")

    console.print(Panel.fit(
        "[bold green]✓ Data preparation complete![/bold green]\n\n"
        f"Files in: [bold]{PROCESSED_DIR}[/bold]\n\n"
        "Next step:\n"
        "  [bold]make finetune[/bold]   — or open notebooks/01_finetune_colab.ipynb"
    ))


if __name__ == "__main__":
    main()
