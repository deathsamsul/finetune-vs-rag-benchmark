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





# # run ragas 
# def run_ragas(dataset, metrics: list, pipeline_name: str ,get_metrics,build_judge,build_embeddings,extract_score) -> dict:
#     """
#     Run RAGAS evaluation with Gemini judge + local embeddings.
#     Returns {metric_name: float_score}.
#     """
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()



#     # Select judge provider and validate API key
#     judge_provider = os.getenv("JUDGE_MODEL", "groq").strip().lower()

#     api_key = None

#     if judge_provider == "local":
#         console.print("[green]Using local judge — no API key required.[/green]")

#     elif judge_provider == "groq":
#         api_key = os.getenv("GROQ_API_KEY", "").strip()

#     elif judge_provider == "gemini":
#         api_key = os.getenv("GEMINI_API_KEY", "").strip()

#     elif judge_provider == "openai":
#         api_key = os.getenv("OPENAI_API_KEY", "").strip()

#     else:
#         raise ValueError(
#             f"Unsupported JUDGE_MODEL: {judge_provider}. "
#             "Choose: local, groq, gemini, or openai."
#         )

#     if judge_provider != "local" and not api_key:
#         raise EnvironmentError(
#             f"{judge_provider.upper()} API key is missing. "
#             f"Set the corresponding API key in your .env file."
#         )



#     faithfulness, answer_relevancy, context_precision = get_metrics()

#     metric_map = {
#         "faithfulness":      faithfulness,
#         "answer_relevancy":  answer_relevancy,
#         "context_precision": context_precision,
#     }
#     selected = [metric_map[m] for m in metrics if m in metric_map]

#     if pipeline_name == "finetuned" and context_precision in selected:
#         selected.remove(context_precision)
#         console.print("  [dim]context_precision skipped for fine-tuned "
#                     "(no retrieval step)[/dim]")

#     judge      = build_judge(api_key)
#     embeddings = build_embeddings()

#     console.print(
#         f"  Running RAGAS ({len(selected)} metrics)  "
#         f"[dim]judge: {judge_provider}  |  embeddings: {EMBED_MODEL}[/dim]"
#     )

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         from ragas import evaluate

#         result = evaluate(
#             dataset,
#             metrics=selected,
#             llm=judge,
#             embeddings=embeddings,
#             raise_exceptions=False,
#         )

#     scores = {}
#     for metric in selected:
#         key   = metric.name
#         score = extract_score(result, key)
#         if score != score:  # NaN check
#             console.print(
#                 f"  [yellow]⚠  {key} = NaN — judge failed. "
#                 f"Check {judge_provider.upper()}_API_KEY quota at {judge_provider.upper()}_API_KEY[/yellow]"
#             )
#         else:
#             console.print(f"  [green]✓[/green]  {key} = [bold]{score:.3f}[/bold]")
#         scores[key] = score

#     return scores




def run_ragas(
    dataset,
    metrics: list,
    pipeline_name: str,
    get_metrics,
    build_judge,
    build_embeddings,
    extract_score,
) -> dict:
    """
    Run RAGAS evaluation with configurable judge and local embeddings.
    Returns {metric_name: float_score}.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Select judge provider and validate API key
    judge_provider = os.getenv("JUDGE_MODEL", "groq").strip().lower()

    api_key = None

    if judge_provider == "local":
        console.print(
            "[green]Using local judge — no API key required.[/green]"
        )

    elif judge_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()

    elif judge_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    elif judge_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

    else:
        raise ValueError(
            f"Unsupported JUDGE_MODEL: {judge_provider}. "
            "Choose: local, groq, gemini, or openai."
        )

    if judge_provider != "local" and not api_key:
        raise EnvironmentError(
            f"{judge_provider.upper()} API key is missing. "
            "Set the corresponding API key in your .env file."
        )

    # Build metrics
    faithfulness, answer_relevancy, context_precision = get_metrics()

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }

    selected = [
        metric_map[m]
        for m in metrics
        if m in metric_map
    ]

    if pipeline_name == "finetuned" and context_precision in selected:
        selected.remove(context_precision)
        console.print(
            "  [dim]context_precision skipped for fine-tuned "
            "(no retrieval step)[/dim]"
        )

    # Free GPU memory before loading judge — the fine-tuned / RAG model
    # is no longer needed (inference already done), and the judge needs VRAM.
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build judge and embeddings
    judge = build_judge(api_key)
    embeddings = build_embeddings()

    console.print(
        f"  Running RAGAS ({len(selected)} metrics) "
        f"[dim]judge: {judge_provider} | embeddings: {EMBED_MODEL}[/dim]"
    )

    # Evaluation configuration
    from ragas import evaluate
    from ragas.run_config import RunConfig

    run_config = RunConfig(
        timeout=900,
        max_retries=3,
        max_workers=1,
    )

    console.print(
        "  [dim]Evaluation config: timeout=900s | "
        "retries=3 | workers=1[/dim]"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = evaluate(
            dataset,
            metrics=selected,
            llm=judge,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )

    # Extract scores
    scores = {}

    for metric in selected:
        key = metric.name
        score = extract_score(result, key)

        if score is None or not isinstance(score, (int, float)) or not __import__("math").isfinite(score):
            console.print(
                f"  [yellow]⚠ {key} = NaN — evaluation failed "
                f"or timed out using {judge_provider} judge.[/yellow]"
            )
        else:
            console.print(
                f"  [green]✓[/green] {key} = [bold]{score:.3f}[/bold]"
            )

        scores[key] = score

    return scores
