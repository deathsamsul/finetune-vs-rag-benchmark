# RAGAS Benchmark Report

Generated: 2026-11-20 09:40

## Setup

| Component | Value |
|---|---|
| LLM judge |  |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Dataset | lavita/ChatDoctor-HealthCareMagic-100k (10k samples) |
| Fine-tune | QLoRA, Llama-3.2-3B-Instruct, 3 epochs |
| RAG | LangChain + FAISS, k=4 chunks |

## Results

| Metric | Fine-Tuned (QLoRA) | RAG (FAISS) | Winner |
|---|---|---|---|
| answer_relevancy | 0.285 | 0.320 | RAG |
| context_precision | N/A | 0.039 | — |
| faithfulness | 0.322 | 0.014 | Fine-Tuned |

## Metric Definitions

| Metric | Measures | Higher = |
|---|---|---|
| faithfulness | Claims in answer supported by context | Less hallucination |
| answer_relevancy | Answer directly addresses the question | More on-topic |
| context_precision | Relevant RAG chunks ranked first (RAG only) | Better retrieval |

## Notes

- Test split held out — no data leakage between pipelines.
- RAGAS judge:  (scores have ~3-5% variance).
- context_precision N/A for fine-tuned (no retrieval step).
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (same model used in RAG pipeline).