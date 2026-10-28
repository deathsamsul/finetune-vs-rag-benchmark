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







# for comparison.py 

def both_score_answers(samples: list[dict],
                  ft_results: list[dict],
                  rag_results: list[dict],
                  console,
                  run_per_sample_ragas) -> list[dict]:
    """Score both pipelines on faithfulness and answer relevancy."""

    from datasets import Dataset

    metric_names = ["faithfulness", "answer_relevancy"]
    all_scores = {}

    for metric_name in metric_names:
        console.print(
            f"\n[bold]Evaluating {metric_name}...[/bold]"
        )

        ft_ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": ft["answer"],
                # Use actual evidence, not the reference answer,
                # for a valid faithfulness evaluation.
                "contexts": ft.get("contexts", []),
                "ground_truth": s["answer"],
            }
            for s, ft in zip(samples, ft_results)
        ])

        rag_ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": rag["answer"],
                "contexts": rag.get("contexts", []),
                "ground_truth": s["answer"],
            }
            for s, rag in zip(samples, rag_results)
        ])

        all_scores[f"ft_{metric_name}"] = (
            run_per_sample_ragas(ft_ds, metric_name)
        )
        all_scores[f"rag_{metric_name}"] = (
            run_per_sample_ragas(rag_ds, metric_name)
        )

    rows = []

    for i, (s, ft, rag) in enumerate(
        zip(samples, ft_results, rag_results)
    ):
        row = {
            "question": s["question"],
            "ground_truth": s["answer"],
            "ft_answer": ft["answer"],
            "rag_answer": rag["answer"],
            "ft_latency_ms": ft.get("latency_ms", 0),
            "rag_latency_ms": rag.get("latency_ms", 0),
            "rag_sources": "|".join(rag.get("sources", [])),
        }

        for metric_name in metric_names:
            ft_score = all_scores[f"ft_{metric_name}"][i]
            rag_score = all_scores[f"rag_{metric_name}"][i]

            row[f"ft_{metric_name}"] = ft_score
            row[f"rag_{metric_name}"] = rag_score
            row[f"{metric_name}_delta"] = ft_score - rag_score

        rows.append(row)

    # Rank samples by the largest difference across either metric.
    rows.sort(
        key=lambda r: max(
            abs(r["faithfulness_delta"]),
            abs(r["answer_relevancy_delta"]),
        ),
        reverse=True,
    )

    return rows





def score_answers(samples: list[dict],
                  ft_results:  list[dict],
                  rag_results: list[dict],
                  metric_name: str,
                  console,
                  run_per_sample_ragas) -> list[dict]:
    """
    Score each answer with RAGAS and return enriched comparison rows.
    Each row: question, ft_answer, rag_answer, ft_score, rag_score, delta.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    metric_map = {
        "faithfulness":     faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    metric = metric_map[metric_name]

    # Build datasets for scoring
    ft_ds = Dataset.from_list([
        {
            "question":     s["question"],
            "answer":       ft["answer"],
            "contexts":     [s["answer"]],      # reference as proxy context
            "ground_truth": s["answer"],
        }
        for s, ft in zip(samples, ft_results)
    ])

    rag_ds = Dataset.from_list([
        {
            "question":     s["question"],
            "answer":       rag["answer"],
            "contexts":     rag.get("contexts", [s["answer"]]),
            "ground_truth": s["answer"],
        }
        for s, rag in zip(samples, rag_results)
    ])

    console.print(f"  Scoring {len(samples)} FT answers with RAGAS ({metric_name})...")
    ft_scores  = run_per_sample_ragas(ft_ds, metric_name)

    console.print(f"  Scoring {len(samples)} RAG answers with RAGAS ({metric_name})...")
    rag_scores = run_per_sample_ragas(rag_ds, metric_name)

    rows = []
    for i, (s, ft, rag) in enumerate(zip(samples, ft_results, rag_results)):
        rows.append({
            "question":        s["question"],
            "ground_truth":    s["answer"],
            "ft_answer":       ft["answer"],
            "rag_answer":      rag["answer"],
            "ft_score":        ft_scores[i],
            "rag_score":       rag_scores[i],
            "delta":           ft_scores[i] - rag_scores[i],   # positive = FT wins
            "ft_latency_ms":   ft.get("latency_ms", 0),
            "rag_latency_ms":  rag.get("latency_ms", 0),
            "rag_sources":     "|".join(rag.get("sources", [])),
        })

    # Sort by absolute delta descending — biggest disagreements first
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows