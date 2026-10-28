import json
import os
import time
from pathlib import Path
from rich.console import Console





console = Console()



#  Fine-Tuned Inference 
class FineTunedInference:

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model     = None
        self._tokenizer = None

    def load(self):
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_dir}\n"
                "Copy adapter files from Kaggle Output into checkpoints/medquad-lora/"
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

        except ImportError:
            console.print("  [yellow]Unsloth not found — using vanilla PEFT loader[/yellow]")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            with open(self.checkpoint_dir / "adapter_config.json") as f:
                base_model_name = json.load(f)["base_model_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.checkpoint_dir),
                clean_up_tokenization_spaces=False,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float16,
                device_map={"": 0},
                token=os.getenv("HF_TOKEN"),
            )
            model = PeftModel.from_pretrained(base, str(self.checkpoint_dir))
            model.eval()

        self._model     = model
        self._tokenizer = tokenizer
        console.print("  [green]✓ Fine-tuned model loaded[/green]")
        return self

    def generate(self, question: str) -> dict:
        from utils.finetuning.utils import format_for_inference
        import torch

        prompt = format_for_inference(question)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

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
        n_in   = inputs["input_ids"].shape[1]
        answer = self._tokenizer.decode(
            outputs[0][n_in:], skip_special_tokens=True
        ).strip()
        return {"answer": answer, "latency_ms": latency_ms,
                "n_tokens": len(outputs[0]) - n_in}
