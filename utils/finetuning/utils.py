"""
utils.py — Shared utilities for fine-tuning pipeline.

Handles:
- Instruction formatting for Llama chat template
- Dataset loading helpers
- Checkpoint validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from rich.console import Console

console = Console()

# ── Llama-3 Chat Template ------------------------------
# Llama 3.2 Instruct expects this exact format.
LLAMA3_CHAT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{user_message}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{assistant_response}<|eot_id|>"
)

SYSTEM_PROMPT = (
    "You are a knowledgeable and careful medical assistant. "
    "Answer the medical question accurately and concisely, "
    "based only on established medical knowledge. "
    "If you are uncertain, say so clearly rather than guessing."
)


def format_for_llama3(question: str, answer: str) -> str:
    """
    Format a Q&A pair into the Llama-3 chat format.
    
    This is what the model is trained on — the same template must
    be used at inference time for the adapter to work correctly.
    """
    return LLAMA3_CHAT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        user_message=question.strip(),
        assistant_response=answer.strip(),
    )


def format_for_inference(question: str) -> str:
    """
    Format a question for inference (no answer — model completes it).
    Used at evaluation time and in the serving layer.
    """
    # Include everything up to the assistant tag; model fills in the rest
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over a JSONL file, yielding one dict per line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: python data/prepare_data.py"
        )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict]:
    """Load a JSONL file into a list, with optional sample limit."""
    samples = []
    for i, obj in enumerate(iter_jsonl(path)):
        if max_samples and i >= max_samples:
            break
        samples.append(obj)
    return samples


def validate_checkpoint(checkpoint_dir: str | Path) -> bool:
    """
    Check if a LoRA checkpoint directory has the expected files.
    Returns True if valid, False otherwise.
    """
    path = Path(checkpoint_dir)
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not (path / f).exists()]

    # adapter_model.bin is an alternative to safetensors
    if "adapter_model.safetensors" in missing:
        if (path / "adapter_model.bin").exists():
            missing.remove("adapter_model.safetensors")

    if missing:
        console.print(f"[yellow]Checkpoint {path} missing: {missing}[/yellow]")
        return False
    return True


def count_parameters(model) -> tuple[int, int]:
    """Return (trainable_params, total_params) for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total
