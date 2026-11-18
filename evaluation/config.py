"""
evaluation/config.py — Centralized configuration for the evaluation pipeline.

Loads settings from .env, then overrides with CLI arguments.
Priority: CLI args → .env → sensible defaults.

This avoids scattering config logic across multiple files and
ensures every module sees the same resolved settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_PROVIDER = "local"
DEFAULT_JUDGE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
DEFAULT_JUDGE_DEVICE = "auto"
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_MAX_TOKENS = 1024
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Provider → API-key env-var mapping
# ---------------------------------------------------------------------------

PROVIDER_KEY_MAP = {
    "groq":       "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "openai":     "OPENAI_API_KEY",
}

PROVIDER_MODEL_MAP = {
    "groq":       ("GROQ_MODEL",       "llama-3.1-8b-instant"),
    "openrouter": ("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    "gemini":     ("GEMINI_MODEL",     "gemini-2.0-flash-lite"),
    "openai":     ("OPENAI_MODEL",     "gpt-4o-mini"),
}


# ---------------------------------------------------------------------------
# EvalConfig — single source of truth for every configurable knob
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Resolved evaluation configuration (CLI → .env → defaults)."""

    # Judge
    judge_provider: str = DEFAULT_JUDGE_PROVIDER
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_device: str = DEFAULT_JUDGE_DEVICE
    judge_load_in_4bit: bool = True
    judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS

    # API keys (resolved per provider)
    api_key: Optional[str] = None

    # Embeddings
    embed_model: str = DEFAULT_EMBED_MODEL

    # Evaluation
    finetuned_model: str = "./checkpoints/medquad-lora"
    rag_index: str = "./indexes/medquad_faiss"
    test_data: str = "data/processed/test.jsonl"
    metrics: list = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_precision"
    ])
    output: str = "evaluation/results/ragas_scores.json"
    n_samples: Optional[int] = None
    skip_finetuned: bool = False
    skip_rag: bool = False
    k: int = 4
    batch_size: int = 8

    @classmethod
    def from_env_and_args(cls, args=None) -> "EvalConfig":
        """
        Build config: .env first, then CLI args override.

        Args:
            args: Namespace from argparse (may be None for .env-only use).
        """
        cfg = cls()

        # --- .env values ---
        cfg.judge_provider = os.getenv(
            "JUDGE_PROVIDER", DEFAULT_JUDGE_PROVIDER
        ).strip().lower()
        cfg.judge_model = os.getenv("JUDGE_MODEL", "").strip()
        cfg.judge_device = os.getenv("JUDGE_DEVICE", DEFAULT_JUDGE_DEVICE).strip()
        cfg.judge_load_in_4bit = os.getenv(
            "JUDGE_LOAD_IN_4BIT", "true"
        ).strip().lower() in ("true", "1", "yes")
        cfg.judge_temperature = float(
            os.getenv("JUDGE_TEMPERATURE", str(DEFAULT_JUDGE_TEMPERATURE))
        )
        cfg.judge_max_tokens = int(
            os.getenv("JUDGE_MAX_TOKENS", str(DEFAULT_JUDGE_MAX_TOKENS))
        )
        cfg.embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()

        # Resolve provider-specific model if JUDGE_MODEL is empty
        if not cfg.judge_model:
            if cfg.judge_provider in PROVIDER_MODEL_MAP:
                env_var, default = PROVIDER_MODEL_MAP[cfg.judge_provider]
                cfg.judge_model = os.getenv(env_var, default).strip()
            else:
                cfg.judge_model = DEFAULT_JUDGE_MODEL

        # Resolve API key
        if cfg.judge_provider in PROVIDER_KEY_MAP:
            key_env = PROVIDER_KEY_MAP[cfg.judge_provider]
            cfg.api_key = os.getenv(key_env, "").strip() or None

        # --- CLI overrides (take precedence) ---
        if args is not None:
            if getattr(args, "judge_provider", None):
                cfg.judge_provider = args.judge_provider.strip().lower()
            if getattr(args, "judge_model", None):
                cfg.judge_model = args.judge_model.strip()
            if getattr(args, "judge_device", None):
                cfg.judge_device = args.judge_device.strip()
            if getattr(args, "judge_temperature", None) is not None:
                cfg.judge_temperature = args.judge_temperature
            if getattr(args, "judge_max_tokens", None) is not None:
                cfg.judge_max_tokens = args.judge_max_tokens

            # Re-resolve API key if provider changed via CLI
            if getattr(args, "judge_provider", None):
                if cfg.judge_provider in PROVIDER_KEY_MAP:
                    key_env = PROVIDER_KEY_MAP[cfg.judge_provider]
                    cfg.api_key = os.getenv(key_env, "").strip() or None

            # Standard eval args
            for attr in (
                "finetuned_model", "rag_index", "test_data",
                "output", "n_samples", "skip_finetuned", "skip_rag", "k",
                "batch_size",
            ):
                val = getattr(args, attr, None)
                if val is not None:
                    setattr(cfg, attr, val)
            if getattr(args, "metrics", None):
                cfg.metrics = args.metrics

        return cfg

    # --- Validation ---

    def validate(self) -> list[str]:
        """
        Return list of validation error messages (empty = OK).
        Does NOT raise — caller decides how to handle.
        """
        errors = []

        # API providers need a key
        if self.judge_provider != "local" and not self.api_key:
            key_env = PROVIDER_KEY_MAP.get(self.judge_provider, "???_API_KEY")
            errors.append(
                f"{self.judge_provider} judge requires {key_env} "
                f"(set in .env or environment)"
            )

        # Model name should not be empty
        if not self.judge_model:
            errors.append("judge_model is empty — set JUDGE_MODEL or --judge-model")

        # Test data must exist
        if not Path(self.test_data).exists():
            errors.append(
                f"Test data not found: {self.test_data}\n"
                "  Run: python -m data.prepare_data"
            )

        return errors

    def summary_str(self) -> str:
        """Human-readable one-liner for console output."""
        if self.judge_provider == "local":
            return f"local ({self.judge_model}, device={self.judge_device})"
        return f"{self.judge_provider} ({self.judge_model})"
