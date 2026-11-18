"""
tests/test_api.py — Integration tests for the FastAPI endpoints.

Uses mocked model state — no real weights, FAISS index, or API keys needed.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from serving import main as serving_main
from serving.schemas import QueryResponse


# ─────────────────────────────────────────────────────────────────────────────
# Shared fake answer builders
# ─────────────────────────────────────────────────────────────────────────────

def _rag_result(question: str = "Q?") -> dict:
    return {
        "answer":     "Diabetes symptoms include increased thirst and frequent urination.",
        "sources":    ["medqa_000001_chunk0", "medqa_000002_chunk1"],
        "contexts":   ["Type 2 diabetes causes... (chunk text)"],
        "latency_ms": 312.4,
    }


def _ft_result(question: str = "Q?") -> dict:
    return {
        "answer":     "Hypertension is diagnosed by measuring blood pressure above 130/80 mmHg.",
        "latency_ms": 445.2,
        "n_tokens":   48,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Restore app state after every test so tests are independent."""
    orig_rag     = serving_main.state.rag_chain
    orig_ft      = serving_main.state.ft_model
    orig_rt      = serving_main.state.rag_tracer
    orig_ft_tr   = serving_main.state.ft_tracer
    orig_lf      = serving_main.state.langfuse
    yield
    serving_main.state.rag_chain  = orig_rag
    serving_main.state.ft_model   = orig_ft
    serving_main.state.rag_tracer = orig_rt
    serving_main.state.ft_tracer  = orig_ft_tr
    serving_main.state.langfuse   = orig_lf


@pytest.fixture
def mock_rag():
    chain = MagicMock()
    chain.query.side_effect = _rag_result
    serving_main.state.rag_chain  = chain
    serving_main.state.rag_tracer = None
    return chain


@pytest.fixture
def mock_ft():
    model = MagicMock()
    model.generate.side_effect = _ft_result
    serving_main.state.ft_model  = model
    serving_main.state.ft_tracer = None
    return model


@pytest.fixture
def ragas_scores() -> dict:
    return {
        "timestamp": "2025-01-15T10:30:00",
        "n_samples": 100,
        "scores": {
            "finetuned": {"faithfulness": 0.82, "answer_relevancy": 0.78},
            "rag":       {"faithfulness": 0.91, "answer_relevancy": 0.85,
                          "context_precision": 0.74},
        },
        "config": {"k": 4},
    }


@pytest.fixture
def client():
    """Bare TestClient — tests set up state themselves via fixtures."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield  # skip real model loading during tests

    serving_main.app.router.lifespan_context = _noop_lifespan
    serving_main.state.langfuse = None
    with TestClient(serving_main.app, raise_server_exceptions=True) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_ok_when_both_loaded(self, client, mock_rag, mock_ft):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert set(body["models_loaded"]) == {"rag", "finetuned"}
        assert body["index_loaded"] is True

    def test_degraded_when_nothing_loaded(self, client):
        serving_main.state.rag_chain = None
        serving_main.state.ft_model  = None
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["models_loaded"] == []

    def test_rag_only_loaded(self, client, mock_rag):
        serving_main.state.ft_model = None
        resp = client.get("/health")
        body = resp.json()
        assert "rag" in body["models_loaded"]
        assert "finetuned" not in body["models_loaded"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /query — RAG mode
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRAG:
    def test_success(self, client, mock_rag):
        resp = client.post("/query", json={
            "question": "What are the symptoms of Type 2 diabetes?",
            "mode": "rag",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "rag"
        assert "diabetes" in body["answer"].lower()
        assert body["sources"] == ["medqa_000001_chunk0", "medqa_000002_chunk1"]
        assert body["latency_ms"] > 0

    def test_calls_chain_with_exact_question(self, client, mock_rag):
        question = "What causes hypertension in adults?"
        client.post("/query", json={"question": question, "mode": "rag"})
        mock_rag.query.assert_called_once_with(question)

    def test_503_when_not_loaded(self, client):
        serving_main.state.rag_chain = None
        resp = client.post("/query", json={
            "question": "What is diabetes mellitus?",
            "mode": "rag",
        })
        assert resp.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# POST /query — Fine-tuned mode
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryFinetuned:
    def test_success(self, client, mock_ft):
        resp = client.post("/query", json={
            "question": "How is hypertension diagnosed clinically?",
            "mode": "finetuned",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "finetuned"
        assert "blood pressure" in body["answer"].lower()
        assert body["sources"] is None   # no retrieval step

    def test_calls_generate_with_exact_question(self, client, mock_ft):
        question = "What is the treatment for mild depression?"
        client.post("/query", json={"question": question, "mode": "finetuned"})
        mock_ft.generate.assert_called_once_with(question)

    def test_503_when_not_loaded(self, client):
        serving_main.state.ft_model = None
        resp = client.post("/query", json={
            "question": "What is diabetes mellitus?",
            "mode": "finetuned",
        })
        assert resp.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# POST /query — Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:
    def test_missing_question_422(self, client, mock_rag):
        resp = client.post("/query", json={"mode": "rag"})
        assert resp.status_code == 422

    def test_question_too_short_422(self, client, mock_rag):
        resp = client.post("/query", json={"question": "Hi?", "mode": "rag"})
        assert resp.status_code == 422

    def test_invalid_mode_422(self, client):
        resp = client.post("/query", json={
            "question": "What is diabetes mellitus?",
            "mode": "gpt4",
        })
        assert resp.status_code == 422

    def test_k_out_of_range_422(self, client, mock_rag):
        resp = client.post("/query", json={
            "question": "What is diabetes mellitus?",
            "mode": "rag",
            "k": 999,
        })
        assert resp.status_code == 422

    def test_whitespace_question_stripped_then_fails(self, client, mock_rag):
        resp = client.post("/query", json={"question": "    ", "mode": "rag"})
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    SCORES_PATH = Path("evaluation/results/ragas_scores.json")

    def test_404_when_no_results_file(self, client):
        """Scores file doesn't exist → 404. Temporarily rename if present."""
        backup = self.SCORES_PATH.with_suffix(".json.bak")
        renamed = False
        if self.SCORES_PATH.exists():
            self.SCORES_PATH.rename(backup)
            renamed = True
        try:
            resp = client.get("/metrics")
            assert resp.status_code == 404
        finally:
            if renamed:
                backup.rename(self.SCORES_PATH)

    def test_returns_scores_when_file_exists(self, client, ragas_scores):
        self.SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.SCORES_PATH.write_text(json.dumps(ragas_scores))
        try:
            resp = client.get("/metrics")
            assert resp.status_code == 200
            body = resp.json()
            assert body["n_samples"] == 100
            assert "rag" in body["scores"]
        finally:
            self.SCORES_PATH.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GET /
# ─────────────────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_returns_200_with_links(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "docs"    in body
        assert "health"  in body
        assert "metrics" in body
