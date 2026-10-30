# RAGAS Benchmark Report

Generated: 2026-10-28 05:54

## Setup

| Component | Value |
|---|---|
| LLM judge |  |
| Embeddings |  (local) |
| Dataset | lavita/ChatDoctor-HealthCareMagic-100k (10k samples) |
| Fine-tune | QLoRA, Llama-3.2-3B-Instruct, 3 epochs |
| RAG | LangChain + FAISS, k=4 chunks |

## Results

| Metric | Fine-Tuned (QLoRA) | RAG (FAISS) | Winner |
|---|---|---|---|
| answer_relevancy | 0.683 | 0.516 | Fine-Tuned |
| context_precision | N/A | 1.000 | — |
| faithfulness | 0.179 | 0.322 | RAG |

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
- Embeddings:  (same model used in RAG pipeline).