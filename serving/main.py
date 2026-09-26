"""
main.py — FastAPI serving layer for both pipelines.

Day 3 script. Serves both the fine-tuned model and the RAG pipeline
via a unified REST API with:
    - Pydantic-validated request/response models
    - Langfuse tracing on every call
    - Health check and metrics endpoints
    - Async request handling

Start with:
    uvicorn serving.main:app --reload --port 8000

Endpoints:
    POST /query           — single question, choose pipeline via 'mode'
    POST /batch-query     — multiple questions in one call
    GET  /health          — model load status
    GET  /metrics         — RAGAS benchmark scores from last eval run
    GET  /docs            — Swagger UI (auto-generated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from serving.schemas import (
    BatchQueryRequest, BatchQueryResponse,
    HealthResponse, MetricsResponse,
    QueryRequest, QueryResponse,
)

load_dotenv()
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))
logger = logging.getLogger(__name__)

# ── App state — shared across requests ────────────────────────────────────────
class AppState:
    rag_chain    = None
    ft_model     = None
    rag_tracer   = None
    ft_tracer    = None
    langfuse     = None


state = AppState()


# ── Lifespan — model loading on startup ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("Loading models...")

    # ── Langfuse client ────────────────────────────────────────────────────────
    try:
        from tracing.langfuse_setup import (
            get_langfuse_client, LangfuseTracer,
        )
        state.langfuse = get_langfuse_client()
        if state.langfuse:
            state.rag_tracer = LangfuseTracer(state.langfuse, pipeline_name="rag")
            state.ft_tracer  = LangfuseTracer(state.langfuse, pipeline_name="finetuned")
    except Exception as e:
        logger.warning(f"Langfuse setup failed: {e} — running without tracing")

    # ── RAG chain ──────────────────────────────────────────────────────────────
    rag_index_path = os.getenv("FAISS_INDEX_PATH", "./indexes/medquad_faiss")
    try:
        from rag.rag_chain import RAGChain
        state.rag_chain = RAGChain(index_path=rag_index_path)
        state.rag_chain.load()
        logger.info("RAG chain loaded ✓")
    except FileNotFoundError:
        logger.warning(
            f"FAISS index not found at {rag_index_path}. "
            "RAG mode will return 503. Run: python rag/build_index.py"
        )
    except Exception as e:
        logger.error(f"RAG chain failed to load: {e}")

    # ── Fine-Tuned model ───────────────────────────────────────────────────────
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints/medquad-lora")
    try:
        from evaluation.run_ragas import FineTunedInference
        state.ft_model = FineTunedInference(checkpoint_dir)
        state.ft_model.load()
        logger.info("Fine-tuned model loaded ✓")
    except FileNotFoundError:
        logger.warning(
            f"Checkpoint not found at {checkpoint_dir}. "
            "Fine-tuned mode will return 503. Run: python finetuning/train_lora.py"
        )
    except Exception as e:
        logger.error(f"Fine-tuned model failed to load: {e}")

    logger.info("Startup complete.")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    if state.langfuse:
        state.langfuse.flush()
    logger.info("Shutdown complete.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fine-Tuned vs. RAG — Medical Q&A API",
    description=(
        "Serves both a QLoRA fine-tuned Llama-3.2-3B and a LangChain+FAISS RAG pipeline "
        "on MedQuAD medical questions. Every call is traced in Langfuse."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware — request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check which models are loaded and ready."""
    loaded = []
    if state.rag_chain is not None:
        loaded.append("rag")
    if state.ft_model is not None:
        loaded.append("finetuned")

    status = "ok" if loaded else "degraded"
    return HealthResponse(
        status=status,
        models_loaded=loaded,
        index_loaded=state.rag_chain is not None,
        message=None if loaded else "No models loaded. Check startup logs.",
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
async def get_metrics():
    """
    Return the RAGAS benchmark scores from the last evaluation run.
    Re-run with: python evaluation/run_ragas.py
    """
    scores_path = Path("evaluation/results/ragas_scores.json")
    if not scores_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run: python evaluation/run_ragas.py",
        )
    with open(scores_path) as f:
        data = json.load(f)
    return MetricsResponse(
        timestamp=data.get("timestamp"),
        n_samples=data.get("n_samples"),
        scores=data.get("scores", {}),
        config=data.get("config"),
    )


@app.post("/query", response_model=QueryResponse, tags=["Inference"])
async def query(req: QueryRequest):
    """
    Answer a medical question using the specified pipeline.
    
    - **mode=rag** (default): retrieves relevant chunks from FAISS, then generates.
    - **mode=finetuned**: uses the QLoRA model directly, no retrieval.
    """
    if req.mode == "rag":
        if state.rag_chain is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not loaded. Run: python rag/build_index.py",
            )
        # Run in a thread pool — model inference is CPU/GPU-bound, not async-friendly
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: state.rag_chain.query(req.question)
        )

        # Trace with Langfuse
        trace_data = {}
        if state.rag_tracer:
            contexts_preview = "\n---\n".join(result.get("contexts", [])[:2])[:400]
            prompt_repr = f"[CONTEXT]\n{contexts_preview}\n[Q] {req.question}"
            trace_data = state.rag_tracer.trace_generation(
                question=req.question,
                prompt=prompt_repr,
                answer=result["answer"],
                latency_ms=result["latency_ms"],
                n_tokens={
                    "input":  len(prompt_repr.split()),
                    "output": len(result["answer"].split()),
                },
                metadata={"sources": result.get("sources", []), "k": req.k},
            )

        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            mode="rag",
            latency_ms=result["latency_ms"],
            sources=result.get("sources"),
            trace_id=trace_data.get("trace_id"),
        )

    elif req.mode == "finetuned":
        if state.ft_model is None:
            raise HTTPException(
                status_code=503,
                detail="Fine-tuned model not loaded. Run: python finetuning/train_lora.py",
            )
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: state.ft_model.generate(req.question)
        )

        trace_data = {}
        if state.ft_tracer:
            from finetuning.utils import format_for_inference
            prompt_repr = format_for_inference(req.question)
            trace_data = state.ft_tracer.trace_generation(
                question=req.question,
                prompt=prompt_repr,
                answer=result["answer"],
                latency_ms=result["latency_ms"],
                n_tokens={
                    "input":  len(prompt_repr.split()),
                    "output": result.get("n_tokens", 0),
                },
            )

        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            mode="finetuned",
            latency_ms=result["latency_ms"],
            sources=None,
            trace_id=trace_data.get("trace_id"),
        )

    # Should not be reachable — Pydantic validates the Literal
    raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")


@app.post("/batch-query", response_model=BatchQueryResponse, tags=["Inference"])
async def batch_query(req: BatchQueryRequest):
    """
    Answer multiple questions in a single call (max 50).
    Each question is processed sequentially to avoid OOM.
    """
    start_total = time.perf_counter()
    results = []

    for question in req.questions:
        single_req = QueryRequest(question=question, mode=req.mode, k=req.k)
        result = await query(single_req)
        results.append(result)

    total_latency = (time.perf_counter() - start_total) * 1000
    return BatchQueryResponse(results=results, total_latency_ms=total_latency)


@app.get("/", tags=["System"])
async def root():
    """Redirect hint to Swagger UI."""
    return JSONResponse({
        "message": "Fine-Tuned vs. RAG Medical Q&A API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    })
