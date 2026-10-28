"""
rag/ — Retrieval-Augmented Generation package.

Public API (lazy imports — heavy deps only loaded when actually used):
    from rag.rag_chain import RAGChain, build_rag_chain
    from rag.embeddings import get_embeddings
"""

__all__ = ["RAGChain", "build_rag_chain", "get_embeddings"]


def __getattr__(name: str):
    if name in ("RAGChain", "build_rag_chain"):
        from evaluation.utils.rag.rag_chain import RAGChain, build_rag_chain  # noqa: F401
        return {"RAGChain": RAGChain, "build_rag_chain": build_rag_chain}[name]
    if name == "get_embeddings":
        from evaluation.utils.rag.embeddings import get_embeddings  # noqa: F401
        return get_embeddings
    raise AttributeError(f"module 'rag' has no attribute {name!r}")
