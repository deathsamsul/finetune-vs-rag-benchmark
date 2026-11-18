"""
utils/args.py — CLI argument parsers for all entry points.

Each parser adds judge configuration flags so the user can switch
providers and models from the terminal without editing Python code.

Priority: CLI args → .env → defaults (handled by EvalConfig).
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Shared: adds judge-related arguments to any parser
# ---------------------------------------------------------------------------

def _add_judge_args(parser: argparse.ArgumentParser):
    """Add --judge-provider, --judge-model, etc. to a parser."""
    judge = parser.add_argument_group("Judge configuration")
    judge.add_argument(
        "--judge-provider", type=str, default=None,
        help="Judge provider: local, groq, openrouter, gemini, openai "
             "(default: JUDGE_PROVIDER env or 'local')",
    )
    judge.add_argument(
        "--judge-model", type=str, default=None,
        help="Judge model name/path "
             "(default: JUDGE_MODEL env or provider-specific default)",
    )
    judge.add_argument(
        "--judge-device", type=str, default=None,
        help="Device for local judge: auto, cpu, cuda (default: auto)",
    )
    judge.add_argument(
        "--judge-temperature", type=float, default=None,
        help="Judge temperature (default: 0)",
    )
    judge.add_argument(
        "--judge-max-tokens", type=int, default=None,
        help="Judge max output tokens (default: 1024)",
    )
    return parser


# ---------------------------------------------------------------------------
# CLI for run_ragas.py
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on fine-tuned and RAG pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--finetuned_model", default="./checkpoints/medquad-lora")
    parser.add_argument("--rag_index",       default="./indexes/medquad_faiss")
    parser.add_argument("--test_data",       default="data/processed/test.jsonl")
    parser.add_argument("--metrics", nargs="+",
                        default=["faithfulness", "answer_relevancy", "context_precision"])
    parser.add_argument("--output",          default="evaluation/results/ragas_scores.json")
    parser.add_argument("--n_samples",       type=int, default=None)
    parser.add_argument("--skip_finetuned",  action="store_true")
    parser.add_argument("--skip_rag",        action="store_true")
    parser.add_argument("--k",               type=int, default=4)
    parser.add_argument("--batch_size",      type=int, default=8,
                        help="Batch size for inference")
    _add_judge_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI for build_index.py
# ---------------------------------------------------------------------------

def build_index_parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from MedQuAD corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, default="data/processed/",
                        help="Directory containing corpus.jsonl")
    parser.add_argument("--index_path", type=str, default="./indexes/medquad_faiss",
                        help="Output path for FAISS index")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Max characters per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Characters overlap between chunks")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Embedding batch size (reduce if OOM)")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limit docs for quick test (None = all)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI for train_lora.py
# ---------------------------------------------------------------------------

def fine_tune_parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning on MedQuAD via Unsloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name (overrides config)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to train.jsonl (overrides config)")
    parser.add_argument("--val_dataset", type=str, default=None,
                        help="Path to val.jsonl (overrides config)")
    parser.add_argument("--method", choices=["lora", "qlora"], default="qlora",
                        help="Fine-tuning method")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Checkpoint output directory (overrides config)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (for quick tests)")
    parser.add_argument("--config", type=str, default="finetuning/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate config & data then exit without training")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI for data preparation
# ---------------------------------------------------------------------------

def data_parse_args():
    HF_DATASET_NAME = os.getenv('HF_DATASET_NAME', "").strip()

    parser = argparse.ArgumentParser(description="Prepare MedQuAD dataset splits")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to use (None = all)")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dataset",     type=str,   default=HF_DATASET_NAME,
                        help="HuggingFace dataset name")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI for compare.py
# ---------------------------------------------------------------------------

def compare_parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fine-tuned vs RAG pipeline on disagreement cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_data",       type=str, default="data/processed/test.jsonl")
    parser.add_argument("--finetuned_model", type=str, default="./checkpoints/medquad-lora")
    parser.add_argument("--rag_index",       type=str, default="./indexes/medquad_faiss")
    parser.add_argument("--n_samples",       type=int, default=50,
                        help="Number of test questions to compare")
    parser.add_argument("--top_n",           type=int, default=10,
                        help="Show top N most divergent cases")
    parser.add_argument("--metric",          type=str, default="faithfulness",
                        choices=["faithfulness", "answer_relevancy"],
                        help="Which per-sample metric to rank by")
    parser.add_argument("--output_csv",      type=str,
                        default="evaluation/results/disagreements.csv")
    _add_judge_args(parser)
    return parser.parse_args()
