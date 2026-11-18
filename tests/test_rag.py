"""
tests/test_rag.py — Unit tests for RAG pipeline components.

All tests are fully self-contained: they mock away langchain/faiss/
sentence-transformers so the suite runs without those heavy deps installed.
On a machine with the full requirements.txt installed, the same tests pass
with real imports (the mocks simply short-circuit expensive I/O).

Run with:
    pytest tests/test_rag.py -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — install lightweight stubs for heavy optional deps
# ─────────────────────────────────────────────────────────────────────────────

def _stub_langchain():
    """Install minimal langchain stubs so rag modules can be imported."""
    # Only stub if the real package isn't already available
    if "langchain" in sys.modules:
        return
    # Skip stubbing if the real langchain stack is installed
    try:
        import langchain_core.documents  # noqa: F401
        import langchain_text_splitters  # noqa: F401
        return  # real packages available — no stubs needed
    except ImportError:
        pass

    # ── langchain.schema ──────────────────────────────────────────────────────
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema_mod = types.ModuleType("langchain.schema")
    schema_mod.Document = Document
    sys.modules["langchain"] = types.ModuleType("langchain")
    sys.modules["langchain.schema"] = schema_mod

    # ── langchain.text_splitter ───────────────────────────────────────────────
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100,
                     separators=None, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> list[str]:
            if len(text) <= self.chunk_size:
                return [text]
            chunks, start = [], 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
                if start >= len(text) - self.chunk_overlap:
                    break
            return chunks or [text]

    splitter_mod = types.ModuleType("langchain.text_splitter")
    splitter_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"] = splitter_mod

    # ── langchain.prompts ─────────────────────────────────────────────────────
    class PromptTemplate:
        def __init__(self, input_variables=None, template=""):
            self.input_variables = input_variables or []
            self.template = template

        def format(self, **kwargs):
            result = self.template
            for k, v in kwargs.items():
                result = result.replace("{" + k + "}", v)
            return result

    prompts_mod = types.ModuleType("langchain.prompts")
    prompts_mod.PromptTemplate = PromptTemplate
    sys.modules["langchain.prompts"] = prompts_mod

    # ── langchain.chains / community / core stubs ─────────────────────────────
    for mod_name in [
        "langchain.chains",
        "langchain_community",
        "langchain_community.vectorstores",
        "langchain_community.embeddings",
        "langchain_core",
        "langchain_core.documents",
        "langchain_huggingface",
        "langchain_text_splitters",
        "sentence_transformers",
        "faiss",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # RetrievalQA stub
    class RetrievalQA:
        @classmethod
        def from_chain_type(cls, **kwargs):
            return MagicMock()

    sys.modules["langchain.chains"].RetrievalQA = RetrievalQA

    # HuggingFaceEmbeddings stub
    class HuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            pass

    sys.modules["langchain_community.embeddings"].HuggingFaceEmbeddings = HuggingFaceEmbeddings

    # FAISS stub
    class FAISSStub:
        @classmethod
        def from_documents(cls, *a, **kw):
            return cls()
        @classmethod
        def load_local(cls, *a, **kw):
            return cls()
        def save_local(self, *a, **kw):
            pass
        def add_documents(self, *a, **kw):
            pass
        def as_retriever(self, **kwargs):
            return MagicMock()

    sys.modules["langchain_community.vectorstores"].FAISS = FAISSStub

    # transformers / pipeline stub
    for mod_name in ["transformers", "torch", "peft"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# Install stubs before any rag.* import
_stub_langchain()


# ─────────────────────────────────────────────────────────────────────────────
# Now import the real modules under test (with stubs in place)
# ─────────────────────────────────────────────────────────────────────────────

from utils.rag.build_index import docs_to_langchain  # noqa: E402
from utils.rag.embeddings import get_embeddings      # noqa: E402
from utils.rag.rag_chain import RAGChain, MEDICAL_RAG_PROMPT  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Chunking — docs_to_langchain
# ─────────────────────────────────────────────────────────────────────────────

class TestDocsToLangchain:
    def _doc(self, doc_id="doc_001",
             question="What is diabetes?",
             answer="Diabetes is a chronic metabolic condition."):
        return {
            "id": doc_id,
            "question": question,
            "answer": answer,
            "text": f"Q: {question}\nA: {answer}",
        }

    def test_single_short_doc_produces_at_least_one_chunk(self):
        chunks = docs_to_langchain([self._doc()], chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1

    def test_chunk_metadata_contains_doc_id(self):
        chunks = docs_to_langchain([self._doc(doc_id="medqa_042")], 500, 50)
        assert all(c.metadata["doc_id"] == "medqa_042" for c in chunks)

    def test_long_doc_produces_multiple_chunks(self):
        long_ans = "Medical detail. " * 60
        chunks = docs_to_langchain([self._doc(answer=long_ans)], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_metadata_has_chunk_idx(self):
        long_ans = "Medical detail. " * 60
        chunks = docs_to_langchain([self._doc(answer=long_ans)], 200, 20)
        indices = [c.metadata["chunk_idx"] for c in chunks]
        assert indices[0] == 0
        assert sorted(indices) == list(range(len(chunks)))

    def test_multiple_docs_produce_multiple_groups(self):
        docs = [self._doc("doc_001"), self._doc("doc_002")]
        chunks = docs_to_langchain(docs, 500, 50)
        ids = {c.metadata["doc_id"] for c in chunks}
        assert "doc_001" in ids
        assert "doc_002" in ids

    def test_source_field_in_metadata(self):
        chunks = docs_to_langchain([self._doc(doc_id="medqa_007")], 500, 50)
        for c in chunks:
            assert "medqa_007" in c.metadata["source"]

    def test_page_content_not_empty(self):
        chunks = docs_to_langchain([self._doc()], 500, 50)
        for c in chunks:
            assert c.page_content.strip() != ""


# ─────────────────────────────────────────────────────────────────────────────
# Embedding wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestGetEmbeddings:
    def test_returns_same_instance_on_repeated_calls(self):
        """lru_cache must return the exact same object both times."""
        get_embeddings.cache_clear()
        e1 = get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        e2 = get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        assert e1 is e2
        get_embeddings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# RAGChain interface
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGChainInterface:
    def test_query_raises_before_load(self):
        import pytest
        chain = RAGChain(index_path="./fake/path")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            chain.query("What is diabetes?")

    def test_retrieve_only_raises_before_load(self):
        import pytest
        chain = RAGChain(index_path="./fake/path")
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            chain.retrieve_only("What is diabetes?")

    def test_query_returns_expected_keys(self):
        """With an injected mock inner chain, query() returns the right shape."""
        # Build a stub Document using our stub class
        from langchain_core.documents import Document

        chain = RAGChain(index_path="./fake/path")
        mock_lc = MagicMock()
        mock_lc.invoke.return_value = {
            "result": "Diabetes symptoms include thirst and fatigue.",
            "source_documents": [
                Document(
                    page_content="Type 2 diabetes causes increased thirst.",
                    metadata={"source": "medqa_000001_chunk0"},
                )
            ],
        }
        chain._chain = mock_lc

        result = chain.query("What are symptoms of diabetes?")
        assert "answer"     in result
        assert "sources"    in result
        assert "contexts"   in result
        assert "latency_ms" in result
        assert result["answer"] == "Diabetes symptoms include thirst and fatigue."
        assert result["sources"] == ["medqa_000001_chunk0"]

    def test_query_latency_is_positive(self):
        chain = RAGChain(index_path="./fake/path")
        mock_lc = MagicMock()
        mock_lc.invoke.return_value = {"result": "Some answer.", "source_documents": []}
        chain._chain = mock_lc

        result = chain.query("Test medical question please?")
        assert result["latency_ms"] > 0

    def test_empty_source_documents_gives_empty_lists(self):
        chain = RAGChain(index_path="./fake/path")
        mock_lc = MagicMock()
        mock_lc.invoke.return_value = {"result": "Answer.", "source_documents": []}
        chain._chain = mock_lc

        result = chain.query("What is hypertension?")
        assert result["sources"]  == []
        assert result["contexts"] == []


# ─────────────────────────────────────────────────────────────────────────────
# RAG prompt template
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPromptTemplate:
    def test_template_contains_required_variables(self):
        assert "context"  in MEDICAL_RAG_PROMPT.input_variables
        assert "question" in MEDICAL_RAG_PROMPT.input_variables

    def test_template_formats_correctly(self):
        formatted = MEDICAL_RAG_PROMPT.format(
            context="Diabetes is a metabolic disease.",
            question="What is diabetes?",
        )
        assert "Diabetes is a metabolic disease." in formatted
        assert "What is diabetes?" in formatted

    def test_template_contains_faithfulness_instruction(self):
        """Prompt must tell the model to answer from context only."""
        assert "ONLY" in MEDICAL_RAG_PROMPT.template
