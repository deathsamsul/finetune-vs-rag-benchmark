"""
utils/fine_tune_inference.py — Fine-tuned model inference engine.

Loads QLoRA adapter via Unsloth (if available) or vanilla PEFT.
Supports CPU and GPU inference without crashing on machines without CUDA.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class FineTunedInference:
    """QLoRA fine-tuned model for single-question inference."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Public alias for internal model."""
        return self._model

    @property
    def tokenizer(self):
        """Public alias for internal tokenizer."""
        return self._tokenizer

    def load(self) -> "FineTunedInference":
        """Load model weights and tokenizer into memory."""
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_dir}\n"
                "Copy adapter files into checkpoints/medquad-lora/ or run fine-tuning."
            )
        console.print(f"  Loading fine-tuned model from {self.checkpoint_dir} ...")

        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.checkpoint_dir),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)

        except (ImportError, Exception):
            console.print("  [yellow]Unsloth not available — using vanilla PEFT loader[/yellow]")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            adapter_config_path = self.checkpoint_dir / "adapter_config.json"
            if not adapter_config_path.exists():
                raise FileNotFoundError(f"Missing adapter_config.json in {self.checkpoint_dir}")

            with open(adapter_config_path, encoding="utf-8") as f:
                base_model_name = json.load(f)["base_model_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.checkpoint_dir),
                clean_up_tokenization_spaces=False,
            )

            # Detect hardware and set appropriate precision and device map
            if torch.cuda.is_available():
                dtype = torch.float16
                device_map = {"": 0}
            else:
                dtype = torch.float32
                device_map = None

            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=dtype,
                device_map=device_map,
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base, str(self.checkpoint_dir))
            model.eval()

        # Avoid max_length vs max_new_tokens conflict warning
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        self._model = model
        self._tokenizer = tokenizer
        console.print("  [green]✓ Fine-tuned model loaded[/green]")
        return self

    def generate(self, question: str) -> dict:
        """Generate response for a question and return answer with metadata."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call .load() before generating answers.")

        import torch
        from utils.finetuning.utils import format_for_inference

        prompt = format_for_inference(question)
        inputs = self._tokenizer(prompt, return_tensors="pt")

        device = getattr(self._model, "device", None)
        if device is None and hasattr(self._model, "parameters"):
            try:
                device = next(self._model.parameters()).device
            except StopIteration:
                device = "cpu"

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        n_in = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][n_in:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "latency_ms": latency_ms,
            "n_tokens": len(new_tokens),
        }
