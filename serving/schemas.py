"""
schemas.py — Pydantic request/response models for the FastAPI serving layer.

All API inputs and outputs are validated through these models.
Pydantic ensures:
- Required fields are present
- Types are correct
- Values are within allowed ranges

Day 3: Pydantic validation + FastAPI serving.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Request Models

class QueryRequest(BaseModel):
    """POST /query — run a question through one of the two pipelines."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The medical question to answer",
        examples=["What are the symptoms of Type 2 diabetes?"],
    )
    mode: Literal["finetuned", "rag"] = Field(
        default="rag",
        description=(
            "'rag' — retrieves relevant context from the FAISS index then generates.\n"
            "'finetuned' — uses the QLoRA-fine-tuned model directly (no retrieval)."
        ),
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="(RAG mode only) Number of chunks to retrieve. Default 4.",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace")
        return stripped

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the main symptoms of Type 2 diabetes?",
                    "mode": "rag",
                    "k": 4,
                }
            ]
        }
    }


class BatchQueryRequest(BaseModel):
    """POST /batch-query — run multiple questions in one request."""

    questions: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of medical questions (max 50 per batch)",
    )
    mode: Literal["finetuned", "rag"] = Field(default="rag")
    k: int = Field(default=4, ge=1, le=20)

    @field_validator("questions")
    @classmethod
    def questions_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v]
        if any(len(q) < 5 for q in cleaned):
            raise ValueError("Each question must be at least 5 characters")
        return cleaned


# Response Models
class QueryResponse(BaseModel):
    """Response from POST /query."""

    question:   str   = Field(..., description="The original question")
    answer:     str   = Field(..., description="The generated answer")
    mode:       str   = Field(..., description="Pipeline used: 'finetuned' or 'rag'")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")

    # RAG-only fields
    sources: list[str] | None = Field(
        default=None,
        description="Source chunk identifiers (RAG mode only)",
    )
    contexts: list[str] | None = Field(
        default=None,
        description="Retrieved context passages (RAG mode only). Excluded from Swagger to keep it readable.",
        exclude=True,          # don't include in JSON response by default
    )

    # Observability
    trace_id: str | None = Field(
        default=None,
        description="Langfuse trace ID for this call (if tracing enabled)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the main symptoms of Type 2 diabetes?",
                    "answer": "Common symptoms include increased thirst, frequent urination...",
                    "mode": "rag",
                    "latency_ms": 342.7,
                    "sources": ["medqa_000042_chunk0", "medqa_001107_chunk2"],
                    "trace_id": "clxabcdef123",
                }
            ]
        }
    }


class BatchQueryResponse(BaseModel):
    """Response from POST /batch-query."""
    results: list[QueryResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: Literal["ok", "degraded", "error"]
    models_loaded: list[str]
    index_loaded: bool
    message: str | None = None


class MetricsResponse(BaseModel):
    """Response from GET /metrics — RAGAS benchmark scores."""
    timestamp:  str | None
    n_samples:  int | None
    scores:     dict
    config:     dict | None = None
