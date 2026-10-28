from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table





console = Console()  # This creates a Console object that you can use to print colored text and tables.
LABELS = {"finetuned": "Fine-Tuned (QLoRA)", "rag": "RAG (FAISS)"}

# Formats numeric scores to three decimal places.
def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "N/A"

# Compares two scores and selects a winner.
def _winner(a, b) -> str:
    if a is None or b is None:
        return "—"
    return "Fine-Tuned" if a > b else "RAG"

# Combines scores into rows for comparison.
def _rows(results: dict) -> list[tuple]:
    ft = results.get("finetuned", {}).get("scores", {})
    rag = results.get("rag", {}).get("scores", {})
    return [(m, ft.get(m), rag.get(m)) for m in sorted(ft.keys() | rag.keys())]


def print_table(results: dict) -> None:
    table = Table(title="RAGAS Benchmark Results", header_style="bold cyan")
    for col in ("Metric", LABELS["finetuned"], LABELS["rag"], "Winner"):
        table.add_column(col)
    for m, ft, rag in _rows(results):
        table.add_row(m, _fmt(ft), _fmt(rag), _winner(ft, rag))
    console.print(table)



# Creates and saves a Markdown benchmark report.
def write_markdown(results: dict, path: Path) -> None:  # function — saving a report
    lines = [
        "# RAGAS Benchmark Report",
        f"\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\n",
        "## Metric Scores\n",
        "| Metric | Fine-Tuned (QLoRA) | RAG (FAISS) | Winner |",
        "|---|---|---|---|",
        *[f"| {m} | {_fmt(ft)} | {_fmt(rag)} | {_winner(ft, rag)} |" for m, ft, rag in _rows(results)],
        "\n## Latency\n",
        "| Metric | Fine-Tuned | RAG |",
        "|---|---|---|",
    ]
    
    ft_lat = results.get("finetuned", {}).get("latency", {})
    rag_lat = results.get("rag", {}).get("latency", {})
    for m in sorted(ft_lat.keys() | rag_lat.keys()):
        lines.append(f"| {m} | {ft_lat.get(m, '—')} | {rag_lat.get(m, '—')} |")
    lines += [
        "\n## Notes\n",
        "- Faithfulness: fewer hallucinated claims. Answer relevancy: more on-topic answers.",
        "- Context precision and faithfulness apply to RAG only (the fine-tuned model has no retrieval).",
        "- RAGAS uses an LLM judge, so treat scores as directional (±3-5%).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"  [green]✓ Report saved: {path}[/green]")