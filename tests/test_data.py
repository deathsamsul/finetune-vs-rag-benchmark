"""
tests/test_data.py — Unit tests for data preparation utilities.

Tests data cleaning, formatting, and splitting logic
without requiring any downloads or GPU.

Run with:
    pytest tests/test_data.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Import the functions we want to test
# ─────────────────────────────────────────────────────────────────────────────

# Import inline to avoid side effects from top-level code
def import_prepare():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "prepare_data", "data/prepare_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_data"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# clean_sample
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanSample:
    @pytest.fixture(autouse=True)
    def prep(self):
        self.mod = import_prepare()

    def test_valid_sample_passes(self):
        sample = {
            "question": "What are symptoms of diabetes?",
            "answer": "Type 2 diabetes symptoms include increased thirst, frequent urination, fatigue.",
        }
        result = self.mod.clean_sample(sample)
        assert result is not None
        assert result["question"] == sample["question"].strip()

    def test_answer_too_short_dropped(self):
        sample = {"question": "What is diabetes?", "answer": "It's bad."}
        assert self.mod.clean_sample(sample) is None

    def test_answer_too_long_dropped(self):
        sample = {"question": "What is diabetes?", "answer": "x" * 2001}
        assert self.mod.clean_sample(sample) is None

    def test_bad_phrase_dropped(self):
        sample = {
            "question": "What should I do?",
            "answer": "As an AI language model, I cannot provide medical advice.",
        }
        assert self.mod.clean_sample(sample) is None

    def test_question_too_short_dropped(self):
        sample = {"question": "Hi?", "answer": "Hello there, this is a valid answer text."}
        assert self.mod.clean_sample(sample) is None

    def test_whitespace_stripped(self):
        sample = {
            "question": "  What is hypertension?  ",
            "answer": "  High blood pressure is a condition where the force of blood against artery walls is too high.  ",
        }
        result = self.mod.clean_sample(sample)
        assert result is not None
        assert not result["question"].startswith(" ")
        assert not result["answer"].endswith(" ")


# ─────────────────────────────────────────────────────────────────────────────
# split_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitDataset:
    @pytest.fixture(autouse=True)
    def prep(self):
        self.mod = import_prepare()

    def make_samples(self, n):
        return [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(n)]

    def test_split_sizes_sum_to_total(self):
        samples = self.make_samples(100)
        train, val, test = self.mod.split_dataset(samples, 0.8, 0.1, seed=42)
        assert len(train) + len(val) + len(test) == 100

    def test_split_ratios_approx(self):
        samples = self.make_samples(1000)
        train, val, test = self.mod.split_dataset(samples, 0.8, 0.1, seed=42)
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100

    def test_deterministic_with_seed(self):
        samples = self.make_samples(100)
        t1, v1, s1 = self.mod.split_dataset(samples, 0.8, 0.1, seed=99)
        t2, v2, s2 = self.mod.split_dataset(samples, 0.8, 0.1, seed=99)
        assert [s["question"] for s in t1] == [s["question"] for s in t2]

    def test_different_seeds_give_different_splits(self):
        samples = self.make_samples(100)
        t1, _, _ = self.mod.split_dataset(samples, 0.8, 0.1, seed=1)
        t2, _, _ = self.mod.split_dataset(samples, 0.8, 0.1, seed=2)
        assert [s["question"] for s in t1] != [s["question"] for s in t2]

    def test_no_overlap_between_splits(self):
        samples = self.make_samples(100)
        train, val, test = self.mod.split_dataset(samples, 0.8, 0.1, seed=42)
        train_qs = {s["question"] for s in train}
        val_qs   = {s["question"] for s in val}
        test_qs  = {s["question"] for s in test}
        assert train_qs.isdisjoint(val_qs)
        assert train_qs.isdisjoint(test_qs)
        assert val_qs.isdisjoint(test_qs)


# ─────────────────────────────────────────────────────────────────────────────
# write_jsonl / iter_jsonl roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonlRoundtrip:
    def test_write_and_read_back(self):
        from utils.finetuning.utils import iter_jsonl

        samples = [
            {"question": "What is diabetes?", "answer": "A metabolic disorder."},
            {"question": "What is hypertension?", "answer": "High blood pressure."},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            tmppath = Path(f.name)
            for s in samples:
                f.write(json.dumps(s) + "\n")

        loaded = list(iter_jsonl(tmppath))
        assert loaded == samples
        tmppath.unlink()

    def test_missing_file_raises(self):
        from utils.finetuning.utils import iter_jsonl
        with pytest.raises(FileNotFoundError):
            list(iter_jsonl("/nonexistent/path/file.jsonl"))
