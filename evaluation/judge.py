"""
evaluation/judge.py — Judge factory for RAGAS evaluation.

Supports:
  - local      : HuggingFace model loaded locally (no API key needed)
  - groq       : Groq cloud API (handles n=1 limitation)
  - openrouter : OpenRouter API (ChatOpenRouter / ChatOpenAI fallback)
  - gemini     : Google Gemini API
  - openai     : OpenAI API

Usage:
    from evaluation.config import EvalConfig
    from evaluation.judge import build_judge, build_embeddings

    cfg = EvalConfig.from_env_and_args(args)
    judge = build_judge(cfg)          # Ragas LLM wrapper
    embeddings = build_embeddings(cfg)  # Ragas Embeddings wrapper
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from rich.console import Console

# Use direct base imports to avoid DeprecationHelper warnings from Ragas 0.2+
try:
    from ragas.llms.base import LangchainLLMWrapper
except ImportError:
    from ragas.llms import LangchainLLMWrapper

try:
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
except ImportError:
    from ragas.embeddings import LangchainEmbeddingsWrapper

if TYPE_CHECKING:
    from evaluation.config import EvalConfig

console = Console()

SUPPORTED_PROVIDERS = ["local", "groq", "openrouter", "gemini", "openai"]


# ---------------------------------------------------------------------------
# Local judge — loads a HuggingFace model, no API key needed
# ---------------------------------------------------------------------------

def _build_local_judge(cfg: "EvalConfig"):
    """
    Load a local HuggingFace model as the RAGAS judge.

    Supports 4-bit quantization for low-VRAM environments (RTX 2050 / Kaggle T4).
    The model is wrapped in a HuggingFacePipeline for LangChain / RAGAS compatibility.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        from langchain_community.llms import HuggingFacePipeline

    model_name = cfg.judge_model
    console.print(f"  Loading local judge: [bold]{model_name}[/bold]")

    # Resolve device
    if cfg.judge_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.judge_device
    console.print(f"  Device: {device}")

    # Build model kwargs
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    # Check if model is already pre-quantized in its config/weights
    is_pre_quantized = any(tag in model_name.lower() for tag in ["bnb-4bit", "4bit", "awq", "gptq"])

    if cfg.judge_load_in_4bit and device != "cpu":
        if is_pre_quantized:
            console.print("  Pre-quantized model detected — loading via device_map='auto'")
            model_kwargs["device_map"] = "auto"
        else:
            console.print("  Loading in 4-bit (bitsandbytes NF4)")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
    elif device == "cpu":
        model_kwargs["dtype"] = torch.float32
        model_kwargs["device_map"] = None
        console.print("  [dim]CPU mode — float32, no quantization[/dim]")
    else:
        model_kwargs["dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    # Try loading with Unsloth first (faster), fall back to vanilla transformers
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.judge_max_tokens + 512,
            dtype=None,
            load_in_4bit=cfg.judge_load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        console.print("  [dim]Loaded via Unsloth[/dim]")
    except (ImportError, Exception):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs,
        )
        model.eval()
        console.print("  [dim]Loaded via transformers[/dim]")

    # Avoid max_length vs max_new_tokens conflict warning
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    pipeline_kwargs = {}
    if device == "cpu":
        pipeline_kwargs["device"] = -1

    gen_kwargs = {
        "max_new_tokens": cfg.judge_max_tokens,
        "return_full_text": False,
    }
    # Only set temperature if sampling; avoid warning when do_sample is False
    if cfg.judge_temperature and cfg.judge_temperature > 0:
        gen_kwargs["temperature"] = cfg.judge_temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **gen_kwargs,
        **pipeline_kwargs,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipe)
    console.print("  [green]✓ Local judge ready[/green]")
    return LangchainLLMWrapper(llm)


# ---------------------------------------------------------------------------
# API judges — each provider returns a LangchainLLMWrapper
# ---------------------------------------------------------------------------

def _build_groq_judge(cfg: "EvalConfig"):
    """Groq cloud judge. Handles Groq's n=1 restriction."""
    from langchain_groq import ChatGroq

    console.print(f"  Using Groq judge: [bold]{cfg.judge_model}[/bold]")

    class SafeChatGroq(ChatGroq):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("n", None)
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, n=1, **kwargs
            )

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            kwargs.pop("n", None)
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, n=1, **kwargs
            )

    llm = SafeChatGroq(
        model=cfg.judge_model,
        api_key=cfg.api_key,
        temperature=cfg.judge_temperature,
        timeout=120,
        max_retries=3,
        n=1,
    )
    return LangchainLLMWrapper(llm)


def _build_openrouter_judge(cfg: "EvalConfig"):
    """OpenRouter judge — tries langchain-openrouter, falls back to ChatOpenAI."""
    import os

    console.print(f"  Using OpenRouter judge: [bold]{cfg.judge_model}[/bold]")

    try:
        from langchain_openrouter import ChatOpenRouter
        llm = ChatOpenRouter(
            model=cfg.judge_model,
            api_key=cfg.api_key,
            temperature=cfg.judge_temperature,
            timeout=120,
            max_retries=3,
        )
    except ImportError:
        from langchain_openai import ChatOpenAI
        base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip()
        llm = ChatOpenAI(
            model=cfg.judge_model,
            api_key=cfg.api_key,
            base_url=base_url,
            temperature=cfg.judge_temperature,
            timeout=120,
            max_retries=3,
        )
    return LangchainLLMWrapper(llm)


def _build_gemini_judge(cfg: "EvalConfig"):
    """Google Gemini judge."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    console.print(f"  Using Gemini judge: [bold]{cfg.judge_model}[/bold]")

    llm = ChatGoogleGenerativeAI(
        model=cfg.judge_model,
        google_api_key=cfg.api_key,
        temperature=cfg.judge_temperature,
        timeout=120,
        max_retries=3,
        convert_system_message_to_human=True,
    )
    return LangchainLLMWrapper(llm)


def _build_openai_judge(cfg: "EvalConfig"):
    """OpenAI judge."""
    from langchain_openai import ChatOpenAI

    console.print(f"  Using OpenAI judge: [bold]{cfg.judge_model}[/bold]")

    llm = ChatOpenAI(
        model=cfg.judge_model,
        api_key=cfg.api_key,
        temperature=cfg.judge_temperature,
        timeout=120,
        max_retries=3,
    )
    return LangchainLLMWrapper(llm)


# ---------------------------------------------------------------------------
# Public API — build_judge / build_embeddings
# ---------------------------------------------------------------------------

def _get_builder(provider: str):
    """Dynamically resolve the builder function (permits testing/mocking)."""
    import sys
    mod = sys.modules[__name__]
    fn_name = f"_build_{provider}_judge"
    return getattr(mod, fn_name, None)


def build_judge(cfg: "EvalConfig"):
    """
    Build and return a RAGAS-compatible LLM judge based on config.

    Returns a LangchainLLMWrapper ready to pass to ragas.evaluate().

    Raises:
        ValueError: if the provider is unknown.
        SystemExit: if a required API key is missing.
    """
    provider = cfg.judge_provider
    builder = _get_builder(provider)

    if builder is None:
        raise ValueError(
            f"Unknown judge provider: '{provider}'. "
            f"Supported: {SUPPORTED_PROVIDERS}"
        )

    # Validate API key for non-local providers
    if provider != "local" and not cfg.api_key:
        from evaluation.config import PROVIDER_KEY_MAP
        key_env = PROVIDER_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
        console.print(
            f"[red]✗ {key_env} not set — required for {provider} judge[/red]\n"
            f"  Set it in .env or pass via environment."
        )
        raise SystemExit(1)

    return builder(cfg)


def build_embeddings(cfg: "EvalConfig"):
    """
    Build RAGAS-compatible embeddings (always local, no API key needed).

    Uses the same embedding model as the RAG pipeline for consistency.
    """
    console.print(f"  Loading embeddings: [dim]{cfg.embed_model}[/dim]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        hf = HuggingFaceEmbeddings(
            model_name=cfg.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return LangchainEmbeddingsWrapper(hf)
