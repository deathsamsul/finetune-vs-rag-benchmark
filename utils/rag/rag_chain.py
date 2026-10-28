"""
rag/rag_chain.py — LangChain RAG pipeline (retrieve → generate).

Key fixes vs previous version:
1. Defaults to local checkpoint (checkpoints/medquad-lora) — no 5.5GB re-download
2. CPU-safe device handling — no disk-offload crash on 8GB RAM machines
3. Uses 'dtype' not deprecated 'torch_dtype'
"""
from __future__ import annotations
import os
import time
from pathlib import Path

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from rich.console import Console

from utils.rag.embeddings import get_embeddings

console = Console()

MEDICAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable medical assistant. Answer the question below
using ONLY the information provided in the context. If the context does not contain
enough information to answer accurately, say "I don't have enough information in
the provided context to answer this."

Context:
{context}

Question: {question}

Answer:""",
)

DEFAULT_K = 4


class RAGChain:

    def __init__(self, index_path: str, model_name: str = None, k: int = DEFAULT_K):
        self.index_path = Path(index_path)
        # Default: local checkpoint — avoids re-downloading 5.5GB model
        self.model_name = model_name or os.getenv(
            "CHECKPOINT_DIR", "./checkpoints/medquad-lora"
        )
        self.k      = k
        self._chain = None
        self._retriever = None

    def load(self) -> "RAGChain":
        # ── FAISS index ------------------------------------------------
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}\n"
                "Run: python run_experiment.py --skip_data --skip_eval"
            )
        embeddings = get_embeddings()
        vs = FAISS.load_local(
            str(self.index_path), embeddings,
            allow_dangerous_deserialization=True,
        )
        self._retriever = vs.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        console.print(f"  ✓ FAISS index loaded from {self.index_path}")

        # ── LLM -------------------------------------------------------------
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFacePipeline

        model_path = str(self.model_name)
        console.print(f"  Loading LLM: [bold]{model_path}[/bold]")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=os.getenv("HF_TOKEN"),
            clean_up_tokenization_spaces=False,
        )

        if torch.cuda.is_available():
            # GPU path — auto device map, float16
            console.print("  [dim]GPU detected — using float16[/dim]")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True,
            )
            pipeline_device = 0
        else:
            # CPU path — do NOT use device_map="auto" (offloads to disk → crash)
            console.print(
                "  [dim]No GPU — loading on CPU (avoids disk-offload crash)[/dim]"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,  # float32 on CPU
                device_map=None,      # explicitly None — no disk offload
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True,
            )
            pipeline_device = -1      # -1 = CPU for pipeline()

        model.eval()

        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device,   # int only — not a device_map string
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipe)
        console.print("  ✓ LLM loaded")

        # ── Chain ---------------------------------------------------
        self._chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self._retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": MEDICAL_RAG_PROMPT},
        )
        console.print("[green]✓ RAG chain ready[/green]")
        return self

    def query(self, question: str) -> dict:
        if self._chain is None:
            raise RuntimeError("Call .load() before .query()")
        t0 = time.perf_counter()
        result = self._chain.invoke({"query": question})
        latency_ms = (time.perf_counter() - t0) * 1000
        source_docs = result.get("source_documents", [])
        return {
            "answer":     result.get("result", "").strip(),
            "sources":    [d.metadata.get("source", "unknown") for d in source_docs],
            "contexts":   [d.page_content for d in source_docs],
            "latency_ms": latency_ms,
        }

    def retrieve_only(self, question: str) -> list:
        if self._retriever is None:
            raise RuntimeError("Call .load() before .retrieve_only()")
        return self._retriever.invoke(question)


def build_rag_chain(index_path: str, k: int = DEFAULT_K) -> RAGChain:
    return RAGChain(index_path=index_path, k=k).load()










