"""
embeddings.py — Embedding model wrapper for RAG pipeline.

Uses sentence-transformers locally (no API key, no cost).
The same embeddings are used for both indexing and query time.
"""

from __future__ import annotations
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console

console = Console()

#  Model choice rationale 
# all-MiniLM-L6-v2:
#   - 22M params, ~80MB download
#   - 384-dim embeddings
#   - ~14k tokens/sec on CPU
#   - Good semantic similarity for medical text
#   - Free, fully local
#
# Alternative: "BAAI/bge-base-en-v1.5" — slightly better MTEB score, 2× slower
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.
    
    lru_cache(1) ensures the embedding model is only loaded once
    per process — loading is slow (~2s) and the model is large (~80MB).
    """
    console.print(f"  Loading embedding model: [bold]{model_name}[/bold]")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )
