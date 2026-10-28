from pathlib import Path
from datetime import datetime 
from utils.fine_tune_inference import console
import os
from rich.table import Table


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
EMBED_MODEL  = os.getenv('EMBED_MODEL',"").strip()






def generate_markdown_report(results: dict, output_path: Path):
    ft  = results.get("finetuned", {})
    rag = results.get("rag",       {})
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt(v):
        if v is None: return "N/A"
        if v != v:    return "NaN"
        return f"{v:.3f}"

    lines = [
        "# RAGAS Benchmark Report",
        f"\nGenerated: {now}",
        "\n## Setup",
        "\n| Component | Value |",
        "|---|---|",
        f"| LLM judge | {GEMINI_MODEL} |",
        f"| Embeddings | {EMBED_MODEL} (local) |",
        "| Dataset | lavita/ChatDoctor-HealthCareMagic-100k (10k samples) |",
        "| Fine-tune | QLoRA, Llama-3.2-3B-Instruct, 3 epochs |",
        "| RAG | LangChain + FAISS, k=4 chunks |",
        "\n## Results",
        "\n| Metric | Fine-Tuned (QLoRA) | RAG (FAISS) | Winner |",
        "|---|---|---|---|",
    ]
    for m in sorted(set(ft) | set(rag)):
        fv = ft.get(m); rv = rag.get(m)
        ok = fv and fv == fv and rv and rv == rv
        w  = ("Fine-Tuned" if fv > rv else "RAG") if ok else "—"
        lines.append(f"| {m} | {fmt(fv)} | {fmt(rv)} | {w} |")

    lines += [
        "\n## Metric Definitions",
        "\n| Metric | Measures | Higher = |",
        "|---|---|---|",
        "| faithfulness | Claims in answer supported by context | Less hallucination |",
        "| answer_relevancy | Answer directly addresses the question | More on-topic |",
        "| context_precision | Relevant RAG chunks ranked first (RAG only) | Better retrieval |",
        "\n## Notes",
        "\n- Test split held out — no data leakage between pipelines.",
        f"- RAGAS judge: {GEMINI_MODEL} (scores have ~3-5% variance).",
        "- context_precision N/A for fine-tuned (no retrieval step).",
        f"- Embeddings: {EMBED_MODEL} (same model used in RAG pipeline).",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"  [green]✓ Report saved: {output_path}[/green]")



#  Reporting ----------------------------------------------------------------

def print_results_table(results: dict):
    table = Table(title="RAGAS Benchmark Results",
                  header_style="bold cyan", show_lines=True)
    table.add_column("Metric",              style="bold",    min_width=22)
    table.add_column("Fine-Tuned (QLoRA)", justify="center", min_width=18)
    table.add_column("RAG (FAISS)",        justify="center", min_width=14)
    table.add_column("Winner",             justify="center", min_width=10)

    ft  = results.get("finetuned", {})
    rag = results.get("rag",       {})

    def fmt(v):
        if v is None: return "N/A"
        if v != v:    return "NaN ⚠"
        return f"{v:.3f}"

    for m in sorted(set(ft) | set(rag)):
        fv  = ft.get(m)
        rv  = rag.get(m)
        ok  = fv is not None and fv == fv and rv is not None and rv == rv
        win = ("🏆 FT" if fv > rv else "🏆 RAG") if ok else "—"
        table.add_row(m, fmt(fv), fmt(rv), win)

    console.print(table)