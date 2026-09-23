"""
train_lora.py — QLoRA Fine-Tuning with Unsloth + W&B Tracking

Day 1 script. Fine-tunes Llama-3.2-3B-Instruct on MedQuAD using QLoRA.

Usage:
    python finetuning/train_lora.py                        # uses config.yaml defaults
    python finetuning/train_lora.py --epochs 1             # quick sanity-check run
    python finetuning/train_lora.py --max_samples 500      # faster iteration

Outputs:
    checkpoints/medquad-lora/          model adapter weights
    wandb/                             training curves (auto-synced)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

# - Lazy imports (heavy — only load when actually training) ----------------------------------------

def load_unsloth():
    """Import Unsloth — provides helpful error if not installed."""
    try:
        from unsloth import FastLanguageModel
        return FastLanguageModel
    except ImportError:
        console.print(
            "[bold red]Unsloth not found.[/bold red] Install it with:\n"
            "  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
            "Or open notebooks/01_finetune_colab.ipynb in Google Colab.",
            style="red"
        )
        sys.exit(1)


def load_config(config_path: str = "finetuning/config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
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


def format_instruction(sample: dict, template: str) -> str:
    """
    Format a Q&A pair into an instruction-following prompt.
    
    Expected keys in sample: 'question', 'answer'
    Returns the formatted string for the model.
    """
    return template.format(
        question=sample["question"].strip(),
        answer=sample["answer"].strip(),
    )


def load_jsonl_dataset(path: str, max_samples: int = None) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    import jsonlines
    path = Path(path)
    if not path.exists():
        console.print(f"[red]Dataset not found: {path}[/red]")
        console.print("Run: python data/prepare_data.py")
        sys.exit(1)

    samples = []
    with jsonlines.open(path) as reader:
        for i, obj in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            samples.append(obj)

    console.print(f"  Loaded [bold]{len(samples):,}[/bold] samples from {path}")
    return samples


def prepare_hf_dataset(samples: list[dict], template: str):
    """Convert list of dicts to a HuggingFace Dataset with formatted text."""
    from datasets import Dataset

    formatted = [
        {"text": format_instruction(s, template)}
        for s in samples
    ]
    return Dataset.from_list(formatted)


def get_trainer(model, tokenizer, train_dataset, val_dataset, cfg: dict,
                output_dir: str, report_to: str):
    """Build and return a SFTTrainer."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    t = cfg["training"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        weight_decay=t["weight_decay"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type=t["lr_scheduler_type"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        evaluation_strategy="steps",
        report_to=report_to,
        seed=t["seed"],
        run_name="medquad-qlora",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=t["dataset_num_proc"],
        packing=t["packing"],
        args=training_args,
    )
    return trainer


def print_training_summary(cfg: dict, args, n_train: int, n_val: int):
    """Print a rich summary table before training starts."""
    table = Table(title="Training Configuration", show_header=True,
                  header_style="bold cyan")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="green")

    t = cfg["training"]
    m = cfg["model"]
    l = cfg["lora"]

    effective_batch = (t["per_device_train_batch_size"] *
                       t["gradient_accumulation_steps"])

    rows = [
        ("Model", args.model or m["name"]),
        ("Method", "QLoRA (4-bit base + LoRA adapter)"),
        ("LoRA rank (r)", str(l["r"])),
        ("LoRA alpha", str(l["lora_alpha"])),
        ("Epochs", str(args.epochs or t["num_train_epochs"])),
        ("Learning rate", str(args.lr or t["learning_rate"])),
        ("Effective batch size", str(effective_batch)),
        ("Max seq length", str(m["max_seq_length"])),
        ("Train samples", f"{n_train:,}"),
        ("Val samples", f"{n_val:,}"),
        ("Output dir", str(args.output_dir or t["output_dir"])),
    ]
    for name, val in rows:
        table.add_row(name, val)

    console.print(table)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    console.print(Panel.fit(
        "[bold blue]🦙 QLoRA Fine-Tuning — MedQuAD / Llama-3.2-3B[/bold blue]\n"
        "[dim]Day 1 of the Fine-tune vs. RAG benchmark[/dim]"
    ))

    # ─ Resolve final config values (CLI overrides YAML) ------------------------------------
    model_name   = args.model      or cfg["model"]["name"]
    train_path   = args.dataset    or cfg["dataset"]["train_path"]
    val_path     = args.val_dataset or cfg["dataset"]["val_path"]
    output_dir   = args.output_dir or cfg["training"]["output_dir"]
    epochs       = args.epochs     or cfg["training"]["num_train_epochs"]
    lr           = args.lr         or cfg["training"]["learning_rate"]
    max_samples  = args.max_samples or cfg["dataset"]["max_samples"]
    load_in_4bit = (args.method == "qlora") and cfg["model"]["load_in_4bit"]
    report_to    = "none" if args.no_wandb else cfg["training"]["report_to"]
    template     = cfg["dataset"]["instruction_template"]

    cfg["training"]["num_train_epochs"] = epochs
    cfg["training"]["learning_rate"] = lr

    #  Load Data ─===-----------------------------------------------------------------------
    console.print("\n[bold]Step 1/4 — Loading datasets...[/bold]")
    train_samples = load_jsonl_dataset(train_path, max_samples)
    val_samples   = load_jsonl_dataset(val_path)

    print_training_summary(cfg, args, len(train_samples), len(val_samples))

    if args.dry_run:
        console.print("\n[yellow]--dry_run specified. Exiting before training.[/yellow]")
        return

    #  Load Model + Tokenizer ─==========================----------------------------------------
    console.print("\n[bold]Step 2/4 — Loading model & applying LoRA...[/bold]")
    FastLanguageModel = load_unsloth()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=load_in_4bit,
        token=os.getenv("HF_TOKEN"),
    )

    # Apply LoRA adapter
    l = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=l["r"],
        target_modules=l["target_modules"],
        lora_alpha=l["lora_alpha"],
        lora_dropout=l["lora_dropout"],
        bias=l["bias"],
        use_gradient_checkpointing=l["use_gradient_checkpointing"],
        random_state=l["random_state"],
        use_rslora=l["use_rslora"],
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    console.print(
        f"  Trainable params: [bold green]{trainable:,}[/bold green] / "
        f"{total:,} = [bold]{100 * trainable / total:.2f}%[/bold]"
    )

    #  Prepare HuggingFace Datasets ==========================-------------
    console.print("\n[bold]Step 3/4 — Preparing datasets...[/bold]")
    train_dataset = prepare_hf_dataset(train_samples, template)
    val_dataset   = prepare_hf_dataset(val_samples, template)

    #  W&B Initialisation ---------------------------------------------------
    if report_to == "wandb":
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "finetune-vs-rag-medqa"),
            entity=os.getenv("WANDB_ENTITY"),
            name="medquad-qlora",
            config={
                "model": model_name,
                "method": args.method,
                "epochs": epochs,
                "lr": lr,
                "lora_r": l["r"],
                "lora_alpha": l["lora_alpha"],
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
            },
        )

    #  Train -==-------------------------------------=------------------
    console.print("\n[bold]Step 4/4 — Training...[/bold]")
    trainer = get_trainer(
        model, tokenizer,
        train_dataset, val_dataset,
        cfg, output_dir, report_to
    )

    start_time = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - start_time

    console.print(
        f"\n  [bold green]✓ Training complete[/bold green] in "
        f"[bold]{elapsed / 60:.1f} min[/bold]"
    )
    console.print(f"  Peak GPU memory: {trainer_stats.metrics.get('train_runtime', 'N/A')}")

    #  Save Adapter + Tokenizer ----------============================================================
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"\n  [bold]Adapter saved to:[/bold] {output_dir}")

    # Save training stats for later reference
    stats_path = Path(output_dir) / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "model": model_name,
            "method": args.method,
            "epochs": epochs,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "training_loss": trainer_stats.training_loss,
            "elapsed_seconds": elapsed,
            "wandb_run": wandb.run.url if report_to == "wandb" else None,
        }, f, indent=2)

    console.print(
        Panel.fit(
            f"[bold green]Day 1 complete![/bold green]\n\n"
            f"Adapter: [bold]{output_dir}[/bold]\n"
            f"Training loss: [bold]{trainer_stats.training_loss:.4f}[/bold]\n\n"
            "Next step:\n"
            "  [bold]make rag-index[/bold]   — build the FAISS index (Day 2)"
        )
    )

    if report_to == "wandb":
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
