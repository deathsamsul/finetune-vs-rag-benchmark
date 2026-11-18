.PHONY: help install data finetune rag-index evaluate serve clean

# ── Default target ────────────────────────────────────────────
help:
	@echo "finetune-vs-rag-medqa — available commands:"
	@echo ""
	@echo "  make install      Install all Python dependencies"
	@echo "  make data         Download & prepare MedQuAD splits"
	@echo "  make finetune     Run QLoRA fine-tuning (Day 1)"
	@echo "  make rag-index    Build FAISS RAG index (Day 2)"
	@echo "  make evaluate     Run RAGAS evaluation on both (Day 2)"
	@echo "  make serve        Start FastAPI server (Day 3)"
	@echo "  make clean        Remove generated artefacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Data ──────────────────────────────────────────────────────
data:
	python -m data.prepare_data

# ── Day 1: Fine-Tuning ────────────────────────────────────────
finetune:
	python -m utils.finetuning.train_lora \
		--model unsloth/Llama-3.2-3B-Instruct \
		--dataset data/processed/train.jsonl \
		--val_dataset data/processed/val.jsonl \
		--method qlora \
		--epochs 3 \
		--output_dir ./checkpoints/medquad-lora

# ── Day 2: RAG Index ──────────────────────────────────────────
rag-index:
	python -m utils.rag.build_index \
		--data_path data/processed/ \
		--index_path ./indexes/medquad_faiss

# ── Day 2: Evaluation ─────────────────────────────────────────
evaluate:
	python -m evaluation.run_ragas \
		--finetuned_model ./checkpoints/medquad-lora \
		--rag_index ./indexes/medquad_faiss \
		--test_data data/processed/test.jsonl \
		--metrics faithfulness answer_relevancy context_precision \
		--output evaluation/results/ragas_scores.json

# ── Day 3: Serve ──────────────────────────────────────────────
serve:
	uvicorn serving.main:app --reload --host 0.0.0.0 --port 8000

# ── Cleanup ───────────────────────────────────────────────────
clean:
	rm -rf checkpoints/ indexes/ data/raw/ data/processed/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
