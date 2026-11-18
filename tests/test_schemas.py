"""
tests/test_schemas.py — Unit tests for Pydantic schemas.

These tests run without any GPU or API keys — they validate that
request/response models accept valid inputs and reject invalid ones.

Run with:
    pytest tests/test_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from serving.schemas import (
    QueryRequest,
    QueryResponse,
    BatchQueryRequest,
    HealthResponse,
    MetricsResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
# QueryRequest
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRequest:
    def test_valid_rag_mode(self):
        req = QueryRequest(question="What are the symptoms of diabetes?", mode="rag")
        assert req.mode == "rag"
        assert req.k == 4  # default

    def test_valid_finetuned_mode(self):
        req = QueryRequest(question="What causes hypertension?", mode="finetuned")
        assert req.mode == "finetuned"

    def test_default_mode_is_rag(self):
        req = QueryRequest(question="What causes hypertension?")
        assert req.mode == "rag"

    def test_question_stripped(self):
        req = QueryRequest(question="  What is diabetes?  ")
        assert req.question == "What is diabetes?"

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Why?")  # < 5 chars

    def test_question_empty(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_question_whitespace_only(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="     ")

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="What is diabetes?", mode="gpt4")  # type: ignore

    def test_k_out_of_range_low(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="What is diabetes?", k=0)

    def test_k_out_of_range_high(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="What is diabetes?", k=21)

    def test_k_valid_custom(self):
        req = QueryRequest(question="What is diabetes?", k=8)
        assert req.k == 8


# ─────────────────────────────────────────────────────────────────────────────
# QueryResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryResponse:
    def test_rag_response(self):
        resp = QueryResponse(
            question="What is diabetes?",
            answer="Diabetes is a condition...",
            mode="rag",
            latency_ms=320.5,
            sources=["doc_001_chunk0"],
        )
        assert resp.sources == ["doc_001_chunk0"]
        assert resp.trace_id is None  # default

    def test_finetuned_response_no_sources(self):
        resp = QueryResponse(
            question="What is diabetes?",
            answer="Diabetes is a condition...",
            mode="finetuned",
            latency_ms=450.0,
        )
        assert resp.sources is None

    def test_serialisation_excludes_contexts(self):
        """contexts field is excluded=True — shouldn't appear in JSON output."""
        resp = QueryResponse(
            question="What is diabetes?",
            answer="Diabetes is a condition...",
            mode="rag",
            latency_ms=320.5,
            contexts=["some raw chunk text"],
        )
        d = resp.model_dump()
        assert "contexts" not in d


# ─────────────────────────────────────────────────────────────────────────────
# BatchQueryRequest
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchQueryRequest:
    def test_valid_batch(self):
        req = BatchQueryRequest(
            questions=["What is diabetes?", "What causes hypertension?"],
            mode="rag",
        )
        assert len(req.questions) == 2

    def test_empty_list_invalid(self):
        with pytest.raises(ValidationError):
            BatchQueryRequest(questions=[])

    def test_question_too_short_in_batch(self):
        with pytest.raises(ValidationError):
            BatchQueryRequest(questions=["Hi?", "What is diabetes?"])


# ─────────────────────────────────────────────────────────────────────────────
# HealthResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthResponse:
    def test_ok_status(self):
        resp = HealthResponse(
            status="ok",
            models_loaded=["rag", "finetuned"],
            index_loaded=True,
        )
        assert resp.status == "ok"
        assert resp.message is None

    def test_degraded_status(self):
        resp = HealthResponse(
            status="degraded",
            models_loaded=[],
            index_loaded=False,
            message="No models loaded",
        )
        assert resp.status == "degraded"


# ─────────────────────────────────────────────────────────────────────────────
# MetricsResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsResponse:
    def test_full_metrics(self):
        resp = MetricsResponse(
            timestamp="2025-01-01T12:00:00",
            n_samples=100,
            scores={
                "finetuned": {"faithfulness": 0.82, "answer_relevancy": 0.78},
                "rag":       {"faithfulness": 0.91, "answer_relevancy": 0.85,
                              "context_precision": 0.74},
            },
        )
        assert resp.scores["rag"]["faithfulness"] == 0.91

    def test_empty_scores(self):
        resp = MetricsResponse(timestamp=None, n_samples=None, scores={})
        assert resp.scores == {}
