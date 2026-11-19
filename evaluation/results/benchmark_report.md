# RAGAS Benchmark Report

Generated: 2026-11-19 13:33

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
| answer_relevancy | NaN | 0.436 | — |
| context_precision | N/A | 0.250 | — |
| faithfulness | 0.111 | 0.000 | — |

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