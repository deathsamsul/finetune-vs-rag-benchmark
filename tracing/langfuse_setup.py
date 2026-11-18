"""
langfuse_setup.py — Langfuse observability client + decorators.

Day 3 script. Wraps every LLM call — fine-tuned inference and RAG
generation — with Langfuse tracing to capture:
    - Prompt template sent to the model
    - Completion text returned
    - Token counts (prompt + completion + total)
    - Latency per call (ms)
    - Estimated cost per call (based on token counts)

The key design: decorators wrap existing functions so the serving
layer and evaluation pipeline don't need to be rewritten — just import
the traced versions.

Usage:
    from tracing.langfuse_setup import get_langfuse_client, trace_rag_call, trace_ft_call
    lf = get_langfuse_client()

    @trace_rag_call(lf)
    def my_rag_query(question):
        return rag_chain.query(question)
"""

from __future__ import annotations

import functools
import os
import time
from typing import Callable

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


def get_langfuse_client():
    """
    Create and return a Langfuse client.
    
    Reads credentials from environment:
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        LANGFUSE_HOST       (optional, defaults to cloud.langfuse.com)

    Returns None if credentials are not configured (non-blocking).
    """
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        console.print("  [green]✓ Langfuse client initialised[/green]")
        return client
    except ImportError:
        console.print("  [yellow]Langfuse not installed — tracing disabled[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [yellow]Langfuse init failed: {e} — tracing disabled[/yellow]")
        return None


class LangfuseTracer:
    """
    Stateful tracer that wraps LLM calls with Langfuse spans.
    
    Usage:
        tracer = LangfuseTracer(lf_client, pipeline_name="rag")
        result = tracer.trace_generation(
            question="...",
            prompt="...",
            answer="...",
            latency_ms=342.0,
            n_tokens={"input": 120, "output": 80},
        )
    """

    def __init__(self, client, pipeline_name: str = "unknown"):
        self.client = client
        self.pipeline_name = pipeline_name

    def trace_generation(
        self,
        question: str,
        prompt: str,
        answer: str,
        latency_ms: float,
        n_tokens: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Record a single LLM generation in Langfuse.
        
        Args:
            question:    The user-facing question
            prompt:      The full prompt sent to the model
            answer:      The generated answer
            latency_ms:  End-to-end latency in ms
            n_tokens:    {"input": int, "output": int} token counts
            metadata:    Any additional metadata (sources, chunk ids, etc.)
        
        Returns the input dict augmented with a "trace_id" field.
        """
        if self.client is None:
            return {"trace_id": None}

        try:
            trace = self.client.trace(
                name=f"{self.pipeline_name}-query",
                input={"question": question},
                output={"answer": answer},
                metadata={
                    "pipeline": self.pipeline_name,
                    "latency_ms": latency_ms,
                    **(metadata or {}),
                },
            )

            trace.generation(
                name="llm-call",
                model=os.getenv("BASE_MODEL", "Llama-3.2-3B-Instruct"),
                input=prompt,
                output=answer,
                usage={
                    "input":  (n_tokens or {}).get("input", 0),
                    "output": (n_tokens or {}).get("output", 0),
                    "total":  (n_tokens or {}).get("input", 0) + (n_tokens or {}).get("output", 0),
                },
                latency=latency_ms / 1000,  # Langfuse expects seconds
            )

            self.client.flush()
            return {"trace_id": trace.id}

        except Exception as e:
            console.print(f"  [yellow]Langfuse trace failed: {e}[/yellow]")
            return {"trace_id": None}


def trace_rag_call(tracer: LangfuseTracer) -> Callable:
    """
    Decorator that wraps a RAG query function with Langfuse tracing.
    
    The wrapped function must accept a 'question' str and return a dict
    with at least 'answer' and 'latency_ms' keys.
    
    Usage:
        @trace_rag_call(tracer)
        def my_rag_fn(question: str) -> dict:
            return rag_chain.query(question)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(question: str, **kwargs) -> dict:
            result = fn(question, **kwargs)

            # Build a simplified prompt representation for tracing
            contexts_preview = "\n---\n".join(
                (result.get("contexts") or [])[:2]
            )[:500]
            prompt_repr = (
                f"[SYSTEM] Medical assistant, answer from context only.\n"
                f"[CONTEXT]\n{contexts_preview}\n"
                f"[QUESTION] {question}"
            )

            trace_data = tracer.trace_generation(
                question=question,
                prompt=prompt_repr,
                answer=result.get("answer", ""),
                latency_ms=result.get("latency_ms", 0),
                n_tokens={
                    "input":  len(prompt_repr.split()),     # rough approximation
                    "output": len(result.get("answer", "").split()),
                },
                metadata={"sources": result.get("sources", [])},
            )
            result.update(trace_data)
            return result

        return wrapper
    return decorator


def trace_ft_call(tracer: LangfuseTracer) -> Callable:
    """
    Decorator that wraps a fine-tuned model inference function.
    
    Same interface as trace_rag_call but for the fine-tuned pipeline.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(question: str, **kwargs) -> dict:
            result = fn(question, **kwargs)

            from utils.finetuning.utils import SYSTEM_PROMPT, format_for_inference
            prompt_repr = format_for_inference(question)

            trace_data = tracer.trace_generation(
                question=question,
                prompt=prompt_repr,
                answer=result.get("answer", ""),
                latency_ms=result.get("latency_ms", 0),
                n_tokens={
                    "input":  len(prompt_repr.split()),
                    "output": result.get("n_tokens", len(result.get("answer", "").split())),
                },
            )
            result.update(trace_data)
            return result

        return wrapper
    return decorator
