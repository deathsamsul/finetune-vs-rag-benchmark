"""
tracing/ — Langfuse observability wrappers.

Public API (lazy imports — langfuse only loaded when actually used):
    from tracing.langfuse_setup import get_langfuse_client, LangfuseTracer
    from tracing.langfuse_setup import trace_rag_call, trace_ft_call
"""

__all__ = [
    "get_langfuse_client",
    "LangfuseTracer",
    "trace_rag_call",
    "trace_ft_call",
]


def __getattr__(name: str):
    _exports = ("get_langfuse_client", "LangfuseTracer", "trace_rag_call", "trace_ft_call")
    if name in _exports:
        from tracing import langfuse_setup as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'tracing' has no attribute {name!r}")
