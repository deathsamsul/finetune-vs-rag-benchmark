import json
import os
import time
from pathlib import Path


class FineTunedInference:
    """QLoRA fine-tuned model for single-question inference."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = None
        self.tokenizer = None

    def load(self) -> "FineTunedInference":

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_dir}\n"
                "Run: python finetuning/train_lora.py"
            )
        
        try:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.checkpoint_dir),
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            
            FastLanguageModel.for_inference(self.model)
        except ImportError:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cfg = json.loads((self.checkpoint_dir / "adapter_config.json").read_text())
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
            base = AutoModelForCausalLM.from_pretrained(
                cfg["base_model_name_or_path"],
                torch_dtype=torch.float16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
            )
            self.model = PeftModel.from_pretrained(base, str(self.checkpoint_dir)).eval()
        return self

    def generate(self, question: str) -> dict:
        import torch
        from evaluation.utils.finetuning.utils import format_for_inference

        inputs = self.tokenizer(format_for_inference(question), return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return {
            "answer": self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "latency_ms": latency_ms,
            "n_tokens": len(new_tokens),
        }