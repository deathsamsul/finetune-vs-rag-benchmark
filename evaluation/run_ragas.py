"""
run_ragas.py — RAGAS evaluation for both fine-tuned and RAG pipelines.
Judge      : gemini-2.0-flash-lite  (GEMINI_API_KEY — free tier, works with v1beta)
Embeddings : sentence-transformers/all-MiniLM-L6-v2  (local, free, real vectors)
Why gemini-2.0-flash-lite and not gemini-1.5-flash?
  langchain-google-genai uses v1beta endpoint. gemini-1.5-flash returns 404
  NOT_FOUND on v1beta for some free-tier accounts. gemini-2.0-flash-lite works
  on v1beta free tier and is faster.
Why local embeddings?
  Gemini embedding API requires paid tier. all-MiniLM-L6-v2 is already
  downloaded for the RAG pipeline, runs on CPU, gives real semantic scores.
Usage:
    python -m evaluation.run_ragas --n_samples 2    # quick test
    python -m evaluation.run_ragas --n_samples 100  # full eval
    python -m evaluation.run_ragas --skip_finetuned # RAG only
Outputs:
    evaluation/results/ragas_scores.json
    evaluation/results/benchmark_report.md
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
from rich.panel import Panel
from utils.fine_tune_inference import FineTunedInference
from utils.fine_tune_inference import console 
from utils.args import parse_args
from utils.rag_utils import build_ragas_dataset_rag ,run_ragas
from utils.fine_tune_utils import build_ragas_dataset_finetuned
from utils.report_utils import generate_markdown_report, print_results_table





# python -m evaluation.run_ragas --n_samples 2



load_dotenv()

EMBED_MODEL  = os.getenv('EMBED_MODEL',"").strip()
RESULTS_DIR  = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)




#  RAGAS Core ------------------------------------------------------------------

def _get_metrics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    return faithfulness, answer_relevancy, context_precision



def _build_judge(api_key: str):
    import os
    import warnings
    from ragas.llms import LangchainLLMWrapper

    judge_model = os.getenv("JUDGE_MODEL", "groq").strip().lower()


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if judge_model == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash" ).strip()

            print(f"  Using Gemini judge: {model_name}")

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0,
                timeout=120,
                max_retries=3,convert_system_message_to_human=True
            )

        elif judge_model == "openai":
            from langchain_openai import ChatOpenAI

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o").strip()

            print(f"  Using OpenAI judge: {model_name}")

            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0,
                timeout=120,
                max_retries=3,
            )

        elif judge_model == "groq":
            from langchain_groq import ChatGroq

            model_name = os.getenv(
                "GROQ_MODEL", "openai/gpt-oss-20b"
            ).strip()

            print(f"  Using Groq judge: {model_name}")

            llm = ChatGroq(
                model=model_name,
                api_key=api_key,
                temperature=0,
                timeout=120,
                max_retries=3,
            )

        else:
            raise ValueError(
                f"Unknown JUDGE_MODEL: {judge_model}. "
                "Choose: gemini, openai, or groq."
            )

    return LangchainLLMWrapper(llm)



def _build_embeddings():
    console.print(f"  Loading embeddings [dim]({EMBED_MODEL})[/dim] ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        from ragas.embeddings import LangchainEmbeddingsWrapper

        hf = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return LangchainEmbeddingsWrapper(hf)




def _extract_score(result, name: str) -> float:
    """Extract float score from RAGAS result (handles list or scalar)."""
    val = result[name]
    if isinstance(val, list):
        valid = [float(v) for v in val
                 if v is not None and str(v) not in ("nan", "NaN", "None")]
        return sum(valid) / len(valid) if valid else float("nan")
    return float(val) if val is not None else float("nan")





#  Main ----------------------------------------------------------------

def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold blue]🔬 RAGAS Evaluation — Fine-Tuned vs. RAG[/bold blue]\n"
        f"[dim]Judge: {os.getenv('GROQ_MODEL')}  |  Embeddings: {EMBED_MODEL} (local)[/dim]"
    ))

    import jsonlines
    test_path = Path(args.test_data)
    if not test_path.exists():
        console.print(f"[red]Test data not found: {test_path}[/red]")
        sys.exit(1)

    with jsonlines.open(test_path) as r:
        test_samples = list(r)
    if args.n_samples:
        test_samples = test_samples[:args.n_samples]
        console.print(f"  Using [bold]{args.n_samples}[/bold] samples")
    console.print(f"  Test samples: [bold]{len(test_samples):,}[/bold]")

    results = {}

    #  Fine-Tuned ----------------------------------------------
    if not args.skip_finetuned:
        console.print("\n[bold]━━ Fine-Tuned Pipeline (QLoRA) ━━[/bold]")
        try:
            ft = FineTunedInference(args.finetuned_model)
            ft.load()
            ds = build_ragas_dataset_finetuned(ft, test_samples)
            results["finetuned"] = run_ragas(
                ds, args.metrics, "finetuned",_get_metrics,_build_judge,_build_embeddings,_extract_score
                )
        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping: {e}[/yellow]")
            results["finetuned"] = {}

        # Release the fine-tuned model's VRAM before the RAG pipeline loads its own copy
        import gc, torch
        if "ft" in locals():
            del ft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #  RAG ---------------------------------------------------------------
    if not args.skip_rag:
        console.print("\n[bold]━━ RAG Pipeline (FAISS + LangChain) ━━[/bold]")
        try:
            from utils.rag.rag_chain import RAGChain
            # Pass local checkpoint path to avoid re-downloading 5.5GB
            rag_chain = RAGChain(
                index_path=args.rag_index,
                model_name=str(Path(args.finetuned_model).resolve()),
                k=args.k,
            )
            rag_chain.load()
            ds = build_ragas_dataset_rag(rag_chain, test_samples)
            results["rag"] = run_ragas(ds, args.metrics, "rag",_get_metrics,_build_judge,_build_embeddings,_extract_score)
        except FileNotFoundError as e:
            console.print(f"  [yellow]Skipping: {e}[/yellow]")
            results["rag"] = {}

    #  Save -----------------------------------------------------------------
    console.print()
    print_results_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp":  datetime.now().isoformat(),
            "n_samples":  len(test_samples),
            "scores":     results,
            "config": {
                "finetuned_model": args.finetuned_model,
                "rag_index":       args.rag_index,
                "metrics":         args.metrics,
                "k":               args.k,
                "judge":           os.getenv("JUDGE_MODEL", "").strip(),
                "embeddings":      EMBED_MODEL,
            },
        }, f, indent=2)
    console.print(f"\n  [green]✓ Scores saved: {output_path}[/green]")

    generate_markdown_report(results, RESULTS_DIR / "benchmark_report.md")

    console.print(Panel.fit(
        "[bold green]✓ Evaluation complete![/bold green]\n\n"
        f"Scores : [bold]{output_path}[/bold]\n"
        f"Report : [bold]{RESULTS_DIR / 'benchmark_report.md'}[/bold]"
    ))


if __name__ == "__main__":
    main()