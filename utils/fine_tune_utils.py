from pathlib import Path
from utils.fine_tune_inference import console
from rich.progress import track









def build_ragas_dataset_finetuned(ft_model, test_samples: list):
    from datasets import Dataset
    rows = []
    for sample in track(test_samples, description="  Fine-tuned inference..."):
        try:
            r = ft_model.generate(sample["question"])
            rows.append({
                "question":     sample["question"],
                "answer":       r["answer"],
                "contexts":     [sample["answer"]],
                "ground_truth": sample["answer"],
            })
        except Exception as e:
            console.print(f"  [yellow]FT error: {e}[/yellow]")
            rows.append({"question": sample["question"], "answer": "",
                         "contexts": [""], "ground_truth": sample["answer"]})
    return Dataset.from_list(rows)