import math
from datasets import Dataset
from rich.progress import track


# dataset building + RAGAS scoring


def run_pipeline(pipeline, samples: list[dict]) -> tuple[Dataset, list[float]]:
    """Run a pipeline over the test set; return a RAGAS dataset and per-sample latencies."""
    rows, latencies = [], []
    for s in track(samples, description=f"  {pipeline.name} inference..."):
        try:
            pred = pipeline.predict(s["question"])
            latencies.append(pred.latency_ms)
        except Exception as e:  # keep failed rows so they score low instead of vanishing
            print(f"  Error on sample: {e}")
            pred = type("P", (), {"answer": "", "contexts": [""]})()
        rows.append({
            "question": s["question"],
            "answer": pred.answer,
            "contexts": pred.contexts,
            "ground_truth": s["answer"],
        })
    return Dataset.from_list(rows), latencies


def score(dataset: Dataset, metric_names: list[str]) -> dict[str, float]:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    # evaluate	Runs evaluation on your dataset using the selected metrics.
    #answer_relevancy	Measures how relevant the generated answer is to the question.
    #context_precision	Measures how relevant the retrieved contexts are.
    #faithfulness	Measures whether the answer is supported by the retrieved context.

    available = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }

    selected = {n: available[n] for n in metric_names if n in available}
    df = evaluate(dataset, metrics=list(selected.values())).to_pandas()  #converts the selected metric objects into a list. evaluation results into a pandas DataFrame.
    # Look up by each metric's own .name: it differs from the key in some RAGAS versions
    return {n: float(df[m.name].mean()) for n, m in selected.items()}


def latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {}
    s = sorted(latencies)
    return {
        "avg_ms": round(sum(s) / len(s), 1),
        "p50_ms": round(s[len(s) // 2], 1),
        "p95_ms": round(s[min(int(len(s) * 0.95), len(s) - 1)], 1),
        "max_ms": round(s[-1], 1),
    }