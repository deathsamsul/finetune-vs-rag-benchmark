"""
build_index.py — Chunk corpus + embed + build FAISS index.

script. Reads the corpus.jsonl produced by prepare_data.py,
splits documents into overlapping chunks, embeds them with
sentence-transformers, and saves a FAISS index to disk.

The saved index can be loaded by rag_chain.py for retrieval.

Usage:
    python rag/build_index.py
    python rag/build_index.py --data_path data/processed/ --index_path ./indexes/medquad_faiss
    python rag/build_index.py --chunk_size 300 --chunk_overlap 50

Outputs:
    indexes/medquad_faiss/         FAISS index + docstore (LangChain format)
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from rag.embeddings import get_embeddings

load_dotenv()
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from MedQuAD corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, default="data/processed/",
                        help="Directory containing corpus.jsonl")
    parser.add_argument("--index_path", type=str, default="./indexes/medquad_faiss",
                        help="Output path for FAISS index")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Max characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Characters overlap between chunks")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Embedding batch size (reduce if OOM)")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limit docs for quick test (None = all)")
    return parser.parse_args()


def load_corpus(data_path: str, max_docs: int | None) -> list[dict]:
    """Load corpus.jsonl — each entry has 'id', 'question', 'answer', 'text'."""
    corpus_path = Path(data_path) / "corpus.jsonl"
    if not corpus_path.exists():
        console.print(f"[red]Corpus not found: {corpus_path}[/red]")
        console.print("Run: python data/prepare_data.py")
        raise SystemExit(1)

    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    console.print(f"  Loaded [bold]{len(docs):,}[/bold] documents from corpus")
    return docs


def docs_to_langchain(
    raw_docs: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Convert raw corpus entries to LangChain Documents with chunking.
    
    Strategy:
    - Each Q&A pair becomes its own Document
    - Long answers are split into overlapping chunks
    - Metadata carries the original doc id, question, and source chunk index
    
    Why chunk?
    The embedding model has a max input length (~256 tokens for MiniLM).
    Long answers that exceed this are truncated, losing information.
    Chunking ensures each vector represents a dense, focused passage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    lc_docs = []
    for raw in raw_docs:
        # Full text = question + answer (gives better retrieval context)
        full_text = f"Q: {raw['question']}\nA: {raw['answer']}"
        chunks = splitter.split_text(full_text)

        for idx, chunk in enumerate(chunks):
            lc_docs.append(Document(
                page_content=chunk,
                metadata={
                    "doc_id":   raw["id"],
                    "question": raw["question"],
                    "chunk_idx": idx,
                    "n_chunks":  len(chunks),
                    "source":   f"{raw['id']}_chunk{idx}",
                }
            ))

    return lc_docs


def build_faiss_index(
    documents: list[Document],
    batch_size: int,
    embeddings,
) -> FAISS:
    """
    Build FAISS index from LangChain Documents.
    
    Embeds in batches to avoid OOM on large corpora.
    FAISS uses IndexFlatL2 by default (exact search, no approximation).
    For > 1M chunks, consider IndexIVFFlat for faster approximate search.
    """
    console.print(f"  Embedding {len(documents):,} chunks in batches of {batch_size}...")

    # Split into batches
    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    # Build index from first batch
    first_batch = batches[0]
    console.print("  Building initial index from first batch...")
    index = FAISS.from_documents(first_batch, embeddings)

    # Add remaining batches
    for batch in track(batches[1:], description="  Adding batches..."):
        index.add_documents(batch)

    return index


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold blue]  Building FAISS RAG Index — MedQuAD[/bold blue]\n"
        "[dim]Day 2: Chunk → Embed → Index[/dim]"
    ))

    #  Load corpus 
    console.print("\n[bold]Step 1/4 — Loading corpus...[/bold]")
    raw_docs = load_corpus(args.data_path, args.max_docs)

    #  Chunk 
    console.print("\n[bold]Step 2/4 — Chunking documents...[/bold]")
    lc_docs = docs_to_langchain(raw_docs, args.chunk_size, args.chunk_overlap)
    console.print(
        f"  {len(raw_docs):,} docs → [bold]{len(lc_docs):,}[/bold] chunks "
        f"(avg {len(lc_docs)/len(raw_docs):.1f} chunks/doc)"
    )

    #  Load Embeddings
    console.print("\n[bold]Step 3/4 — Loading embedding model...[/bold]")
    embeddings = get_embeddings()

    #  Build Index 
    console.print("\n[bold]Step 4/4 — Building FAISS index...[/bold]")
    start = time.time()
    index = build_faiss_index(lc_docs, args.batch_size, embeddings)
    elapsed = time.time() - start

    #  Save 
    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    console.print(f"\n  [green]✓ Index saved to: {index_path}[/green]")

    # Save metadata
    meta = {
        "n_documents": len(raw_docs),
        "n_chunks":    len(lc_docs),
        "chunk_size":  args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "index_type": "FAISS-FlatL2",
        "elapsed_seconds": elapsed,
    }
    with open(index_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    console.print(
        Panel.fit(
            f"[bold green]✓ FAISS index built![/bold green]\n\n"
            f"Chunks indexed: [bold]{len(lc_docs):,}[/bold]\n"
            f"Time taken: [bold]{elapsed:.1f}s[/bold]\n"
            f"Index path: [bold]{index_path}[/bold]\n\n"
            "Next step:\n"
            "  [bold]make evaluate[/bold]   — run RAGAS on both pipelines"
        )
    )


if __name__ == "__main__":
    main()
