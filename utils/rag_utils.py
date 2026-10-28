from pathlib import Path
from utils.fine_tune_inference import  console
from rich.progress import track
import warnings
import os


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
EMBED_MODEL  = os.getenv('EMBED_MODEL',"").strip()


#  RAGAS Dataset Builders ----------------------

def build_ragas_dataset_rag(rag_chain, test_samples: list):
    from datasets import Dataset
    rows = []
    for sample in track(test_samples, description="  RAG inference..."):
        try:
            r = rag_chain.query(sample["question"])
            rows.append({
                "question":     sample["question"],
                "answer":       r["answer"],
                "contexts":     r["contexts"] or [""],
                "ground_truth": sample["answer"],
            })
        except Exception as e:
            console.print(f"  [yellow]RAG error: {e}[/yellow]")
            rows.append({"question": sample["question"], "answer": "",
                         "contexts": [""], "ground_truth": sample["answer"]})
    return Dataset.from_list(rows)





# run ragas 
def run_ragas(dataset, metrics: list, pipeline_name: str ,get_metrics,build_judge,build_embeddings,extract_score) -> dict:
    """
    Run RAGAS evaluation with Gemini judge + local embeddings.
    Returns {metric_name: float_score}.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()


    # Validate key
    # api_key = os.getenv("GEMINI_API_KEY", "").strip()
    # if not api_key:
    #     console.print("[red]GEMINI_API_KEY not set in .env[/red]")
    #     console.print("  Free key: https://aistudio.google.com/app/apikey")
    #     raise SystemExit(1)

    
    # Validate Groq API key
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()
    if not api_key:
        console.print("[red]GROQ_API_KEY not set in .env[/red]")
        console.print("  Create a key at: https://console.groq.com/keys")
        raise SystemExit(1)


    faithfulness, answer_relevancy, context_precision = get_metrics()

    metric_map = {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
    }
    selected = [metric_map[m] for m in metrics if m in metric_map]

    if pipeline_name == "finetuned" and context_precision in selected:
        selected.remove(context_precision)
        console.print("  [dim]context_precision skipped for fine-tuned "
                    "(no retrieval step)[/dim]")

    judge      = build_judge(api_key)
    embeddings = build_embeddings()

    console.print(
        f"  Running RAGAS ({len(selected)} metrics)  "
        f"[dim]judge: {GROQ_MODEL}  |  embeddings: {EMBED_MODEL}[/dim]"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas import evaluate

        result = evaluate(
            dataset,
            metrics=selected,
            llm=judge,
            embeddings=embeddings,
            raise_exceptions=False,
        )

    scores = {}
    for metric in selected:
        key   = metric.name
        score = extract_score(result, key)
        if score != score:  # NaN check
            console.print(
                f"  [yellow]⚠  {key} = NaN — judge failed. "
                f"Check GEMINI_API_KEY quota at aistudio.google.com[/yellow]"
            )
        else:
            console.print(f"  [green]✓[/green]  {key} = [bold]{score:.3f}[/bold]")
        scores[key] = score

    return scores
