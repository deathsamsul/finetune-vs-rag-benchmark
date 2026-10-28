import time
from dataclasses import dataclass
from utils.finetuning.inference import FineTunedInference

ALL_METRICS = {"faithfulness", "answer_relevancy", "context_precision"}


# adapters: both pipelines expose the same predict()


@dataclass
class Prediction:
    answer: str
    contexts: list[str]
    latency_ms: float


class FineTunedPipeline:
    name = "finetuned"
    metrics = {"answer_relevancy"}  # no retrieval, so context-based metrics don't apply

    def __init__(self, checkpoint_dir: str):
        self.model = FineTunedInference(checkpoint_dir).load()

    def predict(self, question: str) -> Prediction:
        out = self.model.generate(question)
        return Prediction(out["answer"], [""], out["latency_ms"])


class RAGPipeline:
    name = "rag"
    metrics = ALL_METRICS

    def __init__(self, index_path: str, k: int):
        from utils.rag.rag_chain import RAGChain

        self.chain = RAGChain(index_path=index_path, k=k)
        self.chain.load()

    def predict(self, question: str) -> Prediction:
        t0 = time.perf_counter()
        out = self.chain.query(question)
        return Prediction(out["answer"], out["contexts"], (time.perf_counter() - t0) * 1000)