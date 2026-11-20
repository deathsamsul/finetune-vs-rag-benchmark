# 🏥 Fine-Tuned LLM vs. RAG — Medical Q&A Benchmark

> **A rigorous, reproducible head-to-head evaluation of LoRA fine-tuning vs. RAG on MedQuAD.**  
> Featuring a **Modular, Dynamic LLM Judge Architecture** supporting local 4-bit models (Laptop & Kaggle GPU) and cloud API providers (Groq, OpenRouter, Gemini, OpenAI) with zero code modifications.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-101%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Weights & Biases](https://img.shields.io/badge/Tracked-W%26B-yellow)](https://wandb.ai)
[![Langfuse](https://img.shields.io/badge/Traced-Langfuse-purple)](https://langfuse.com)

---

## 📋 Executive Summary: "Fine-Tune or RAG?"

The question *"Should we fine-tune an open-source LLM or deploy a RAG pipeline?"* is answered with empirical data, latency statistics, and qualitative case analyses.

### 🏆 Head-to-Head Benchmark Snapshot (n = 50 held-out samples)

| Evaluation Metric | Fine-Tuned (QLoRA) | RAG (FAISS + LangChain) | Winner | Key Finding |
|---|:---:|:---:|:---:|---|
| **Faithfulness** ↑ | **0.266** | 0.078 | 🏆 **Fine-Tuned** | FT stays tightly focused on medical prompt definitions; RAG relies on retriever chunk coverage |
| **Answer Relevancy** ↑ | 0.524 | **0.546** | 🏆 **RAG** | RAG generates comprehensive answers addressing clinical questions directly |
| **Context Precision** ↑ | *N/A* (no retriever) | **0.750** | 🏆 **RAG** | FAISS retriever places ground-truth relevant chunks in top-4 rank 75% of the time |
| **Average Latency** ↓ | **~10.5 s** | ~14.8 s | 🏆 **Fine-Tuned** | Fine-tuned has **zero retrieval overhead**; RAG adds chunking + embedding + similarity search |
| **Traceability & Citations** | ❌ Black-box weights | ✅ Source chunk IDs | 🏆 **RAG** | RAG provides traceable medical literature chunks for clinical audit |
| **Knowledge Update Cost** | ❌ Full retraining / LoRA | ✅ Zero-retraining index update | 🏆 **RAG** | New medical guidelines can be inserted in minutes without GPU retraining |

> ⬆ **Higher is better** for RAGAS quality metrics · ⬇ **Lower is better** for latency  
> *Data generated dynamically via `evaluation/run_ragas.py` and visual dashboard in `evaluation/visualize_benchmark.py`.*

---

## 📊 Performance Dashboard & Visual Charts

All benchmark charts are dynamically exported to `evaluation/results/visualizations/` and `visualize-charts/fine-tune-charts/`.

### 1. Aggregate Quality Metrics Comparison
Comparison across RAGAS core evaluation dimensions (Faithfulness, Answer Relevancy, Context Precision):

![Aggregate Metric Comparison](evaluation/results/visualizations/aggregate_metric_comparison.png)

* **Faithfulness**: Measures whether claims in the generated response are factually supported by evidence.
* **Answer Relevancy**: Evaluates semantic alignment with the patient query via reverse-generated question similarity.
* **Context Precision**: Measures how well relevant medical textbook chunks are ranked at the top of the retrieval stack.

---

### 2. Inference Latency & End-to-End Speed
Evaluation timing recorded per query across both pipelines:

![Latency Comparison](evaluation/results/visualizations/latency_comparison.png)

* **Fine-Tuned Pipeline**: Direct single-stage generation (`~10.5s` on T4 GPU).
* **RAG Pipeline**: Two-stage execution (`~14.8s` on T4 GPU) — Vector embedding computation via `all-MiniLM-L6-v2`, FAISS index retrieval ($k=4$), context stuffing, and constrained prompt generation.

---

### 3. Per-Sample Disagreements & Delta Distribution
Where the two approaches disagree most on specific medical cases:

| Disagreement Score Delta | Per-Sample Distribution |
|:---:|:---:|
| ![Score Delta](evaluation/results/visualizations/score_delta.png) | ![Per-Sample Scores](evaluation/results/visualizations/per_sample_score_comparison.png) |

---

### 4. QLoRA Fine-Tuning Performance & System Efficiency
Loss progression and GPU resource utilization tracked via Weights & Biases:

| Training Loss & Validation Curves | System VRAM & GPU Utilization |
|:---:|:---:|
| ![Training Curves](visualize-charts/fine-tune-charts/training-charts.png) | ![System Charts](visualize-charts/fine-tune-charts/system-charts.png) |

* **Base Model**: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
* **Adapter**: LoRA rank $r=16$, $\alpha=32$, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* **Training Convergence**: Train Loss dropped from **2.155** to **1.944** across 423 steps (3 epochs). Eval Loss: **2.068**.

---

## 🔬 Helpfulness & Answer Quality Analysis

### The Tradeoff Matrix

```mermaid
quadrantChart
    title Medical Architecture Tradeoff Matrix
    x-axis Low Latency --> High Latency
    y-axis Static Knowledge --> Dynamic Knowledge & Freshness
    quadrant-1 RAG Pipeline (High Auditability)
    quadrant-2 Hybrid RAG + Fine-Tuning (Enterprise Ideal)
    quadrant-3 Out-of-the-Box Base LLM
    quadrant-4 Fine-Tuned Model (Low Latency / High Fluency)
    "Fine-Tuned (QLoRA)": [0.25, 0.35]
    "RAG (FAISS)": [0.72, 0.88]
    "Hybrid (FT + RAG)": [0.65, 0.92]
```

### When to Choose Which Architecture?

#### Choose Fine-Tuning (QLoRA) if:
1. **Low Latency is Critical**: Instant real-time triage where every millisecond counts.
2. **Domain Tone & Jargon**: You need the model to consistently speak in concise clinical phrasing (USMLE board format) without verbose system prompts.
3. **Closed Knowledge Base**: The domain is stable (e.g. established human anatomy) and doesn't change daily.

#### Choose RAG (FAISS + LangChain) if:
1. **Zero-Hallucination & Auditability**: Doctors/auditors need to see exact citations and book source chunks (`rag_sources`).
2. **Dynamic Knowledge**: Medical guidelines, formulary drugs, and clinic protocols update regularly without GPU retraining.
3. **Cost of Maintenance**: Ingesting a new 500-page medical PDF takes 15 seconds with FAISS chunking vs. hours of GPU fine-tuning.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph DataPrep ["1. Data Preparation (MedQuAD)"]
        Raw[Raw MedQuAD / MedQA] --> Split[Split 80/10/10]
        Split --> Train[train.jsonl - 3,148 Q&As]
        Split --> Val[val.jsonl - 393 Q&As]
        Split --> Test[test.jsonl - 394 Q&As]
        Split --> Corpus[corpus.jsonl - 3,935 docs]
    end

    subgraph PathA ["Path A: Fine-Tuning (QLoRA)"]
        Train --> Unsloth[Unsloth FastLanguageModel]
        Unsloth --> LoRA[LoRA Adapters r=16, a=32]
        LoRA --> FTModel[checkpoints/medquad-lora]
    end

    subgraph PathB ["Path B: RAG Pipeline"]
        Corpus --> Chunker[Text Chunking - 12,438 Chunks]
        Chunker --> Embed[sentence-transformers MiniLM-L6]
        Embed --> FAISS[(FAISS Vector Index)]
        FAISS --> Retriever[Similarity Retriever k=4]
        Retriever --> RAGChain[LangChain RetrievalQA Chain]
    end

    subgraph Eval ["Modular Evaluation & Dynamic Visuals"]
        Test --> Runner[run_ragas.py]
        FTModel --> Runner
        RAGChain --> Runner
        Runner --> Judge{"Modular Judge Selector"}
        Judge -->|Local 4-bit GPU| Qwen[Qwen2.5-7B-Instruct]
        Judge -->|Free Cloud API| Groq[Groq API]
        Judge -->|Cloud API| Gemini[Gemini 2.0 Flash]
        Runner --> Scores[ragas_scores.json]
        Runner --> CSV[disagreements.csv]
        Runner --> AutoVis[visualize_benchmark.py]
        AutoVis --> HTML[benchmark_report.html]
        AutoVis --> Charts[PNG Comparison Charts]
    end
```

---

## ⚡ Execution Guide (Kaggle & Local)

### ❓ Is `!python -m evaluation.run_ragas --n_samples 10` enough?

**YES!** In our updated architecture, `run_ragas.py` is an **all-in-one unified command**:

1. Runs **Fine-Tuned inference** & computes latency.
2. Runs **RAG retrieval & generation** & records source chunks.
3. Completely **unloads inference models from GPU VRAM** to give the judge 100% of GPU resources.
4. Executes **RAGAS scoring** (Faithfulness, Answer Relevancy, Context Precision).
5. Automatically writes `evaluation/results/ragas_scores.json`.
6. Automatically creates `evaluation/results/disagreements.csv`.
7. Automatically compiles `evaluation/results/benchmark_report.md`.
8. Automatically triggers `visualize_benchmark.py` to generate **all PNG comparison charts and the interactive `benchmark_report.html` dashboard**!

### Full Kaggle Notebook Commands

```python
# 1. Set environment variables (Use local GPU judge or Cloud APIs)
import os
os.environ["JUDGE_MODEL"] = "local"
os.environ["LOCAL_JUDGE_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"
os.environ["EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Build FAISS Index (takes ~18 seconds for 12,438 chunks)
!python -m utils.rag.build_index

# 3. Run complete evaluation + automatic chart generation (10 samples)
!python -m evaluation.run_ragas --n_samples 10

# 4. (Optional) Run deep-dive question-by-question disagreement ranking
!python -m evaluation.compare --n_samples 10 --metric faithfulness
```

---

## 🛠️ Tech Stack & Decisions

| Layer | Technology | Decision Rational |
|---|---|---|
| **Fine-Tuning** | Unsloth + QLoRA | ~2× faster training than vanilla PEFT, fits on free T4 (16GB VRAM) |
| **Base LLM** | `Llama-3.2-3B-Instruct` | High instruction compliance; lightweight; ideal for edge and local GPU |
| **Vector DB** | FAISS FlatL2 (Local) | Zero external server dependencies; reproducible; microsecond search |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dimensional dense vectors; free; runs on CPU/GPU with high semantic fidelity |
| **LLM Judge** | Modular (Qwen2.5-7B local / Groq / Gemini) | Zero cost; enables 100% private offline evaluation or fast cloud benchmarking |
| **Evaluation** | RAGAS Framework | Isolates retrieval accuracy from hallucination rate |
| **Serving** | FastAPI + Pydantic v2 | Production-ready schema validation and dual-endpoint routing (`/query`) |
| **Observability** | Langfuse + Weights & Biases | Full token tracking, per-sample latency tracing, and training curve telemetry |

---

## 📂 Project Organization

```
finetune-vs-rag-benchmark/
├── checkpoints/
│   └── medquad-lora/               # LoRA adapter weights, config, and tokenizer
├── data/
│   └── processed/
│       ├── train.jsonl             # 3,148 training pairs
│       ├── val.jsonl               # 393 validation pairs
│       ├── test.jsonl              # 394 held-out evaluation pairs
│       └── corpus.jsonl            # 3,935 RAG knowledge base documents
├── evaluation/
│   ├── run_ragas.py                # Main benchmark execution engine
│   ├── compare.py                  # Side-by-side disagreement analysis
│   ├── visualize_benchmark.py      # Automated chart & HTML dashboard generator
│   ├── scoring.py                  # Metric aggregation & latency calculations
│   ├── providers/
│   │   └── local_provider.py       # 4-bit quantized local judge engine (Qwen2.5-7B)
│   └── results/
│       ├── ragas_scores.json       # Evaluator score output
│       ├── benchmark_report.md     # Markdown benchmark summary
│       ├── disagreements.csv       # Side-by-side question-level comparison
│       └── visualizations/
│           ├── aggregate_metric_comparison.png
│           ├── latency_comparison.png
│           ├── score_delta.png
│           ├── per_sample_score_comparison.png
│           └── benchmark_report.html
├── utils/
│   ├── rag/
│   │   ├── build_index.py          # Chunking & FAISS indexer
│   │   ├── embeddings.py           # Embedding loader
│   │   └── rag_chain.py            # LangChain retrieval chain
│   └── finetuning/
│       ├── train_lora.py           # QLoRA fine-tuning loop
│       └── utils.py                # Chat template formatters
├── visualize-charts/
│   └── fine-tune-charts/           # W&B training telemetry charts
├── serving/
│   ├── main.py                     # FastAPI application
│   └── schemas.py                  # Request/response schemas
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 🚀 Serving API (`/query`)

Launch the unified inference server:

```bash
uvicorn serving.main:app --reload --port 8000
```

### Request Example

```json
POST /query
{
  "question": "What is the primary management for acute adrenal crisis after surgery?",
  "mode": "rag" // or "finetuned"
}
```

### Response Example

```json
{
  "answer": "Immediate administration of IV fluids and stress-dose intravenous hydrocortisone...",
  "mode": "rag",
  "latency_ms": 1342.1,
  "sources": ["medqa_chunk_421", "medqa_chunk_899"]
}
```

---

## 👤 Author

**Samsul Hoque Mondal**  
*MLOps / GenAI / Agentic AI Engineer*  
[LinkedIn](https://linkedin.com/in/deathsamsul) · [GitHub](https://github.com/deathsamsul)
