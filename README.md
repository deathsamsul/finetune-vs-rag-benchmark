# 🏥 Fine-Tuned LLM vs. RAG — Medical Q&A Benchmark

> **A rigorous, reproducible head-to-head evaluation of LoRA fine-tuning vs. RAG on MedQuAD.**
> Same base model. Same domain. Same evaluator. Real numbers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Weights & Biases](https://img.shields.io/badge/Tracked-W%26B-yellow)](https://wandb.ai)
[![Langfuse](https://img.shields.io/badge/Traced-Langfuse-purple)](https://langfuse.com)

---

## 📋 TL;DR

**"Should we fine-tune or use RAG?"** — answered with numbers, not opinions.

| Metric | Fine-Tuned (QLoRA) | RAG (FAISS + LangChain) | Winner |
|---|---|---|---|
| Faithfulness ↑ | — | — | — |
| Answer Relevancy ↑ | — | — | — |
| Context Precision ↑ | N/A | — | — |
| Avg. Latency (ms/query) ↓ | — | — | — |
| Avg. Tokens / query ↓ | — | — | — |
| Hallucination Rate ↓ | — | — | — |

> ⬆ Higher is better for quality metrics · ⬇ Lower is better for cost/latency  
> *Table auto-populated from `evaluation/results/ragas_scores.json` after Day 2 run.*

---

## 🏗️ Architecture

```
MedQuAD Dataset
│
├─── [PATH A: Fine-Tuning] ────────────────────────────────────────────────┐
│    Format as instruction pairs → QLoRA fine-tune Llama-3.2-3B (Unsloth) │
│    → Fine-Tuned Model Checkpoint                                          │
│                                                                            │
└─── [PATH B: RAG] ─────────────────────────────────────────────────────── │
     Chunk + Embed → FAISS Index → LangChain Retrieval Chain               │
     (uses base Llama-3.2-3B, no fine-tuning)                              │
                                                                             │
     Both paths ──→ RAGAS Evaluation (same held-out test split) ───────────┘
                         │
                    ┌────┴─────────────────────┐
                    │  Faithfulness            │
                    │  Answer Relevancy        │
                    │  Context Precision (RAG) │
                    └────────────────────────  ┘
                         │
                    FastAPI Serving Layer (Pydantic-validated)
                    Langfuse Tracing (tokens, latency, cost)
                    W&B Experiment Tracking (training curves)
```

---

## 🛠️ Tech Stack

| Layer | Tool | Why This Choice |
|---|---|---|
| **Fine-tuning** | Unsloth + LoRA/QLoRA | ~2× faster than vanilla PEFT on free T4 GPU |
| **Base model** | `Llama-3.2-3B-Instruct` | Fits on Colab free tier with QLoRA; reproducible |
| **Dataset** | MedQuAD (47k medical Q&A pairs) | Rich, real-world domain; HF-hosted |
| **Retrieval** | LangChain + FAISS | Fully local; no external API keys |
| **Evaluation** | RAGAS | Reference-free; separates hallucination from relevancy |
| **Observability** | Langfuse | Per-call tracing: tokens, latency, cost |
| **Experiment tracking** | Weights & Biases | Training loss curves, hyperparameter sweeps |
| **Serving** | FastAPI + Pydantic | Production-grade validation; unified API for both modes |

---

## 📂 Repo Structure

```
finetune-vs-rag-medqa/
├── data/
│   ├── raw/                        # MedQuAD raw download (auto-fetched)
│   └── processed/
│       ├── train.jsonl             # 80% split — instruction-formatted
│       ├── val.jsonl               # 10% split — early stopping
│       └── test.jsonl             # 10% split — RAGAS eval (held out)
│
├── finetuning/
│   ├── train_lora.py               # Main QLoRA training script (Unsloth)
│   ├── config.yaml                 # All hyperparameters in one place
│   └── utils.py                    # Format helpers, dataset loaders
│
├── rag/
│   ├── build_index.py              # Chunk → embed → FAISS index
│   ├── rag_chain.py                # LangChain retrieval + generation
│   └── embeddings.py               # Embedding model wrapper
│
├── evaluation/
│   ├── run_ragas.py                # Run RAGAS on both pipelines
│   ├── compare.py                  # Side-by-side result comparison
│   └── results/
│       ├── ragas_scores.json       # Raw RAGAS output
│       └── benchmark_report.md    # Auto-generated markdown report
│
├── serving/
│   ├── main.py                     # FastAPI app
│   └── schemas.py                  # Pydantic request/response models
│
├── tracing/
│   └── langfuse_setup.py           # Langfuse client + decorators
│
├── notebooks/
│   └── 01_finetune_colab.ipynb    # Full Colab notebook (Day 1)
│
├── docs/
│   └── langfuse_trace.png         # Trace screenshot (added Day 3)
│
├── .github/
│   └── workflows/
│       └── eval.yml               # CI: run RAGAS on every push
│
├── requirements.txt                # Pinned dependencies
├── requirements-dev.txt            # Linting, testing extras
├── .env.example                    # Required env vars template
├── Makefile                        # One-command shortcuts
└── README.md
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.10+
- GPU with ≥ 8GB VRAM **or** Google Colab free tier (T4)
- Hugging Face account (free) — to access Llama 3.2
- Langfuse account (free tier) — for tracing
- W&B account (free tier) — for experiment tracking

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/finetune-vs-rag-medqa.git
cd finetune-vs-rag-medqa

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — add your actual keys (see .env.example for all vars)
```

### 3. Download & Prepare Data

```bash
python data/prepare_data.py
# Downloads MedQuAD from HuggingFace, creates train/val/test splits
# Output: data/processed/{train,val,test}.jsonl
```

### 4. Fine-Tune with QLoRA (Day 1)

**Option A — Local GPU:**
```bash
python finetuning/train_lora.py \
  --model unsloth/Llama-3.2-3B-Instruct \
  --dataset data/processed/train.jsonl \
  --val_dataset data/processed/val.jsonl \
  --method qlora \
  --epochs 3 \
  --output_dir ./checkpoints/medquad-lora
```

**Option B — Google Colab (recommended for free GPU):**  
Open `notebooks/01_finetune_colab.ipynb` in Colab and run all cells.

### 5. Build RAG Index (Day 2)

```bash
python rag/build_index.py \
  --data_path data/processed/ \
  --index_path ./indexes/medquad_faiss
# Chunks documents, embeds with sentence-transformers, saves FAISS index
```

### 6. Run RAGAS Evaluation (Day 2)

```bash
python evaluation/run_ragas.py \
  --finetuned_model ./checkpoints/medquad-lora \
  --rag_index ./indexes/medquad_faiss \
  --test_data data/processed/test.jsonl \
  --metrics faithfulness answer_relevancy context_precision \
  --output evaluation/results/ragas_scores.json
# Generates evaluation/results/benchmark_report.md automatically
```

### 7. Serve Both Pipelines (Day 3)

```bash
uvicorn serving.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 🔬 Evaluation Methodology

### Why RAGAS?

RAGAS evaluates without requiring hand-labelled gold answers — critical for medical QA where annotation is expensive. It uses an LLM judge internally but the metrics are well-defined:

| Metric | What It Measures | Formula Sketch |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the context? | `|supported claims| / |total claims|` |
| **Answer Relevancy** | Does the answer directly address the question? | Embedding sim of reverse-generated questions |
| **Context Precision** | Are relevant chunks ranked highest in retrieval? | Precision@k with relevance labels |

### Test Set Integrity

- The test split (`data/processed/test.jsonl`) is **held out from fine-tuning entirely**
- Both pipelines are evaluated on identical questions
- No data leakage between paths

### Why These Metrics Isolate the "Why"

- Faithfulness catches **hallucination** — did the model make things up?
- Answer Relevancy catches **off-topic** responses
- Context Precision (RAG only) tells you if retrieval is the bottleneck

---

## 🚀 API Reference

### POST `/query`

```python
# Request
{
  "question": "What are the symptoms of Type 2 diabetes?",
  "mode": "rag"   # or "finetuned"
}

# Response
{
  "answer": "Type 2 diabetes symptoms include...",
  "mode": "rag",
  "latency_ms": 342.7,
  "sources": ["doc_chunk_42", "doc_chunk_107"]   # RAG only
}
```

### GET `/health`

```json
{"status": "ok", "models_loaded": ["finetuned", "rag"]}
```

### GET `/metrics`

Returns the RAGAS benchmark scores from the last evaluation run.

---

## 📊 Observability

Every LLM call — fine-tuned inference and RAG retrieval+generation — is traced in Langfuse:

- **Prompt template** sent to the model
- **Completion** returned
- **Token counts** (prompt + completion + total)
- **Latency** per call
- **Estimated cost** per call

Access your traces at [cloud.langfuse.com](https://cloud.langfuse.com).

---

## 🎯 Design Decisions & Tradeoffs

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Base model | Llama-3.2-3B | Mistral-7B | Fits free Colab T4 with QLoRA |
| Fine-tuning method | QLoRA | Full fine-tune | 10× less VRAM; comparable quality |
| Fine-tuning library | Unsloth | vanilla PEFT | ~2× faster on same hardware |
| Vector DB | FAISS (local) | Pinecone, Weaviate | No API keys; fully local; reproducible |
| Evaluator | RAGAS | Manual labelling | No gold answers needed; separates metrics |
| Embedding model | `all-MiniLM-L6-v2` | OpenAI ada-002 | Free; local; no API calls during indexing |

---

## ⚠️ Limitations

1. **Single domain** — results apply to MedQuAD. Different domains (legal, financial) would need separate runs and may produce different winners.
2. **RAGAS evaluator variance** — it uses an LLM judge internally, so scores have ~3–5% variance between runs. Treat as directional.
3. **No human eval** — the disagreement cases (where fine-tune and RAG give different answers) haven't been human-reviewed yet. That's the next step.
4. **Fine-tuned model coverage** — the QLoRA model can't answer questions outside MedQuAD; the RAG index can be extended by adding documents without retraining.
5. **Colab instability** — free Colab sessions disconnect after ~12h; checkpoint saving every N steps is essential.

---

## 📅 Build Log

- [ ] **Day 1** — Unsloth + QLoRA fine-tune on MedQuAD (Colab). Training loss curve in W&B.
- [ ] **Day 2** — LangChain + FAISS RAG pipeline. RAGAS eval on both. Scores in table above.
- [ ] **Day 3** — Langfuse tracing. Pydantic validation. FastAPI serving. Benchmark report auto-generated.
- [ ] **Day 4** — GitHub push. LinkedIn post with RAGAS scores.

---

## 📝 License

MIT — see [LICENSE](LICENSE)

## 👤 Author

**Samsul Hoque Mondal** — B.Tech CS (MLOps / GenAI /Agentic Ai )  
[LinkedIn](https://linkedin.com/in/deathsamsul) · [GitHub](https://github.com/deathsamsul)
