# RAGAS Benchmark Report

Generated: 2026-11-20 23:27

## Setup

| Component | Value |
|---|---|
| LLM judge | Qwen/Qwen2.5-7B-Instruct (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Dataset | MedQuAD / MedQA (held-out test split) |
| Fine-tune | QLoRA, Llama-3.2-3B-Instruct, 3 epochs |
| RAG | LangChain + FAISS, k=4 chunks |

## Results

| Metric | Fine-Tuned (QLoRA) | RAG (FAISS) | Winner |
|---|---|---|---|
| answer_relevancy | 0.221 | 0.205 | Fine-Tuned |
| context_precision | N/A | 0.053 | — |
| faithfulness | 0.241 | 0.050 | Fine-Tuned |

## Metric Definitions

| Metric | Measures | Higher = |
|---|---|---|
| faithfulness | Claims in answer supported by context | Less hallucination |
| answer_relevancy | Answer directly addresses the question | More on-topic |
| context_precision | Relevant RAG chunks ranked first (RAG only) | Better retrieval |

## Notes

- Test split held out — no data leakage between pipelines.
- RAGAS judge: Qwen/Qwen2.5-7B-Instruct (scores have ~3-5% variance).
- context_precision N/A for fine-tuned (no retrieval step).
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (same model used in RAG pipeline).