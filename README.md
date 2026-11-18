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
| Faithfulness ↑ | 0.266 | 0.078 | Fine-Tuned ⚠️ |
| Answer Relevancy ↑ | 0.524 | 0.546 | RAG |
| Context Precision ↑ | N/A | 0.750 | — |

> ⬆ Higher is better for quality metrics  
> ⚠️ Faithfulness uses different context sources — see [Limitations](#-limitations) and [Experiment Report](docs/EXPERIMENT_REPORT.md)  
> *50 samples evaluated from held-out test set · Judge: Groq · Full results: `evaluation/results/ragas_scores.json`*

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
| **Dataset** | ChatDoctor-HealthCareMagic-100k (medical Q&A pairs) | Rich, real-world domain; HF-hosted |
| **Retrieval** | LangChain + FAISS | Fully local; no external API keys |
| **Evaluation** | RAGAS | Reference-free; separates hallucination from relevancy |
| **Observability** | Langfuse | Per-call tracing: tokens, latency, cost |
| **Experiment tracking** | Weights & Biases | Training loss curves, hyperparameter sweeps |
| **Serving** | FastAPI + Pydantic | Production-grade validation; unified API for both modes |

---

## 📂 Repo Structure

```
finetune-vs-rag-benchmark/
├── data/
│   ├── prepare_data.py              # Download, clean, split dataset
│   ├── raw/                         # Raw download (auto-fetched)
│   └── processed/
│       ├── train.jsonl              # 80% split (7,827 samples)
│       ├── val.jsonl                # 10% split (978 samples)
│       ├── test.jsonl               # 10% split (979 samples) — held out
│       └── corpus.jsonl             # Full corpus for RAG indexing
│
├── utils/
│   ├── finetuning/
│   │   ├── train_lora.py            # Main QLoRA training script (Unsloth)
│   │   ├── config.yaml              # All hyperparameters in one place
│   │   ├── utils.py                 # Llama-3 chat template formatting
│   │   └── inference.py             # Fine-tuned model inference class
│   ├── rag/
│   │   ├── build_index.py           # Chunk → embed → FAISS index
│   │   ├── rag_chain.py             # LangChain retrieval + generation
│   │   └── embeddings.py            # Embedding model wrapper
│   ├── fine_tune_inference.py       # Alternative FT inference class
│   ├── fine_tune_utils.py           # RAGAS dataset builder for FT
│   ├── rag_utils.py                 # RAGAS dataset builder for RAG
│   ├── args.py                      # CLI argument parsers
│   └── report_utils.py              # Report generation utilities
│
├── evaluation/
│   ├── run_ragas.py                 # Run RAGAS on both pipelines
│   ├── compare.py                   # Side-by-side result comparison
│   ├── scoring.py                   # RAGAS scoring utilities
│   ├── pipelines.py                 # Pipeline adapter classes
│   ├── visualize_benchmark.py       # Visualization generator
│   └── results/
│       ├── ragas_scores.json        # Raw RAGAS output (50 samples)
│       └── benchmark_report.md      # Auto-generated markdown report
│
├── serving/
│   ├── main.py                      # FastAPI app
│   └── schemas.py                   # Pydantic request/response models
│
├── tracing/
│   └── langfuse_setup.py            # Langfuse client + decorators
│
├── tests/
│   ├── test_schemas.py              # 21 tests — Pydantic models
│   ├── test_finetuning.py           # 18 tests — FT utils
│   ├── test_data.py                 # 13 tests — data pipeline
│   ├── test_rag.py                  # 16 tests — RAG components
│   └── test_api.py                  # 17 tests — FastAPI endpoints
│
├── docs/
│   ├── PROJECT_AUDIT.md             # Full audit report
│   ├── EXPERIMENT_REPORT.md         # Experimental methodology & results
│   └── HARDWARE_COMPATIBILITY.md    # Hardware testing & Kaggle config
│
├── checkpoints/
│   └── medquad-lora/                # QLoRA adapter checkpoint
│
├── indexes/
│   └── medquad_faiss/               # FAISS index + metadata
│
├── .github/workflows/eval.yml       # CI: run RAGAS on every push
├── requirements.txt                  # Dependencies
├── requirements-dev.txt              # Linting, testing extras
├── .env.example                      # Required env vars template
├── Makefile                          # One-command shortcuts
├── pytest.ini                        # Test configuration
└── README.md
```

---

## ⚡ Quickstart

### Prerequisites

- Python 3.10+
- GPU with ≥ 4GB VRAM (inference) or ≥ 16GB VRAM (training) — or Kaggle T4
- Hugging Face account (free) — to access Llama 3.2
- API key for LLM judge (Groq / Gemini / OpenRouter) — for RAGAS evaluation
- Langfuse account (free tier) — for tracing (optional)
- W&B account (free tier) — for experiment tracking (optional)

### 1. Clone & Install

```bash
git clone https://github.com/deathsamsul/finetune-vs-rag-benchmark.git
cd finetune-vs-rag-benchmark

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
python -m data.prepare_data
# Downloads dataset from HuggingFace, creates train/val/test splits
# Output: data/processed/{train,val,test,corpus}.jsonl
```

### 4. Fine-Tune with QLoRA (Day 1)

**Option A — Local GPU (≥ 16GB VRAM) or Kaggle T4:**
```bash
python -m utils.finetuning.train_lora \
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
python -m utils.rag.build_index \
  --data_path data/processed/ \
  --index_path ./indexes/medquad_faiss
# Chunks documents, embeds with sentence-transformers, saves FAISS index
```

### 6. Run RAGAS Evaluation (Day 2)

```bash
python -m evaluation.run_ragas \
  --finetuned_model ./checkpoints/medquad-lora \
  --rag_index ./indexes/medquad_faiss \
  --test_data data/processed/test.jsonl \
  --metrics faithfulness answer_relevancy context_precision \
  --n_samples 50 \
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
- ⚠️ **Known issue:** The RAG index currently includes test data in its corpus — see [Limitations](#-limitations)

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

## ⚠️ Limitations & Known Issues

### Evaluation Fairness Caveats

> See [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md) and [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md) for full details.

1. **RAG index includes test data** — `corpus.jsonl` contains all splits (train+val+test), so the RAG retriever can find test answers during evaluation. This inflates RAG's context_precision. Fix: rebuild the index using only training data.

2. **Faithfulness uses different context sources** — The fine-tuned model's RAGAS evaluation uses ground-truth answers as context, while RAG uses retrieved document chunks. These measure different things and are not directly comparable.

3. **RAG uses fine-tuned model** — Both approaches use the QLoRA adapter for generation. The comparison is "FT alone vs FT+retrieval", not "FT vs base+retrieval" as originally intended.

### General Limitations

4. **Single domain** — results apply to medical Q&A. Different domains (legal, financial) would need separate runs and may produce different winners.
5. **RAGAS evaluator variance** — it uses an LLM judge internally, so scores have ~3–5% variance between runs. Treat as directional.
6. **No human eval** — the disagreement cases (where fine-tune and RAG give different answers) haven't been human-reviewed yet.
7. **Fine-tuned model coverage** — the QLoRA model can't answer questions outside the training domain; the RAG index can be extended by adding documents without retraining.
8. **Small sample size** — current evaluation uses 50 samples. Full evaluation on 979 test samples recommended for statistical confidence.

---

## 📅 Build Log

- [x] **Day 1** — Unsloth + QLoRA fine-tune on ChatDoctor-HealthCareMagic (Kaggle T4). Training loss: 2.16 (3 epochs, 423 steps).
- [x] **Day 2** — LangChain + FAISS RAG pipeline. RAGAS eval on both (50 samples). Scores in table above.
- [x] **Day 3** — Langfuse tracing. Pydantic validation. FastAPI serving. Benchmark report auto-generated.
- [x] **Day 4** — Full audit, 88/88 tests passing, hardware validation, documentation.

---

## 🖥️ Hardware Requirements

| Task | Minimum GPU VRAM | Recommended |
|---|---|---|
| Inference (4-bit) | 4 GB | RTX 2050+ / T4 |
| RAGAS Evaluation | 4 GB + API key | T4 (faster) |
| QLoRA Training | 16 GB | Kaggle T4 × 2 |
| Full-precision inference | 12 GB | A100 / L4 |

> Tested on RTX 2050 (4 GB VRAM): model loads in ~47s, inference ~13s/query, peak 2.37 GB VRAM.  
> See [docs/HARDWARE_COMPATIBILITY.md](docs/HARDWARE_COMPATIBILITY.md) for full details.

---

## 📄 Documentation

| Document | Description |
|---|---|
| [Project Audit](docs/PROJECT_AUDIT.md) | Full codebase audit, verification checklist, issues found |
| [Experiment Report](docs/EXPERIMENT_REPORT.md) | Evaluation methodology, metrics, fairness caveats |
| [Hardware Compatibility](docs/HARDWARE_COMPATIBILITY.md) | Local testing, resource usage, Kaggle configuration |

---

## 📝 License

MIT — see [LICENSE](LICENSE)

## 👤 Author

**Samsul Hoque Mondal** — B.Tech CS (MLOps / GenAI /Agentic Ai )  
[LinkedIn](https://linkedin.com/in/deathsamsul) · [GitHub](https://github.com/deathsamsul)
