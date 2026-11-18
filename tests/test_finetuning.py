"""
tests/test_finetuning.py — Unit tests for fine-tuning utilities.

Tests prompt formatting and checkpoint validation without
requiring any GPU, model weights, or Unsloth.

Run with:
    pytest tests/test_finetuning.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from utils.finetuning.utils import (
    format_for_llama3,
    format_for_inference,
    load_jsonl,
    validate_checkpoint,
    count_parameters,
    SYSTEM_PROMPT,
    LLAMA3_CHAT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# format_for_llama3
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatForLlama3:
    def test_contains_system_prompt(self):
        result = format_for_llama3("What is diabetes?", "A metabolic disease.")
        assert SYSTEM_PROMPT in result

    def test_contains_question(self):
        result = format_for_llama3("What is diabetes?", "A metabolic disease.")
        assert "What is diabetes?" in result

    def test_contains_answer(self):
        result = format_for_llama3("What is diabetes?", "A metabolic disease.")
        assert "A metabolic disease." in result

    def test_contains_llama3_control_tokens(self):
        result = format_for_llama3("Q?", "A.")
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>" in result
        assert "<|eot_id|>" in result

    def test_strips_whitespace_from_inputs(self):
        result = format_for_llama3("  What is diabetes?  ", "  A disease.  ")
        # Should not have leading/trailing spaces inside the content sections
        assert "  What is diabetes?  " not in result
        assert "What is diabetes?" in result

    def test_user_header_before_question(self):
        result = format_for_llama3("What is diabetes?", "A disease.")
        user_idx     = result.index("user")
        question_idx = result.index("What is diabetes?")
        assert user_idx < question_idx

    def test_assistant_header_before_answer(self):
        result = format_for_llama3("What is diabetes?", "A disease.")
        assistant_idx = result.index("assistant")
        answer_idx    = result.index("A disease.")
        assert assistant_idx < answer_idx


# ─────────────────────────────────────────────────────────────────────────────
# format_for_inference
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatForInference:
    def test_contains_question(self):
        result = format_for_inference("What is hypertension?")
        assert "What is hypertension?" in result

    def test_ends_with_assistant_tag(self):
        result = format_for_inference("What is hypertension?")
        # The prompt ends with the assistant header so the model fills in the rest.
        # strip() removes trailing whitespace/newlines, so we check the raw string.
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_does_not_contain_answer(self):
        result = format_for_inference("What is hypertension?")
        assert "A:" not in result
        assert "Answer:" not in result

    def test_contains_system_prompt(self):
        result = format_for_inference("What is hypertension?")
        assert SYSTEM_PROMPT in result


# ─────────────────────────────────────────────────────────────────────────────
# load_jsonl
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadJsonl:
    def _write_jsonl(self, path: Path, samples: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def test_loads_all_samples(self):
        samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            self._write_jsonl(p, samples)
            loaded = load_jsonl(p)
            assert loaded == samples

    def test_max_samples_respected(self):
        samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            self._write_jsonl(p, samples)
            loaded = load_jsonl(p, max_samples=5)
            assert len(loaded) == 5

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/does_not_exist.jsonl")

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.jsonl"
            p.write_text("")
            loaded = load_jsonl(p)
            assert loaded == []

    def test_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text('{"question":"Q1","answer":"A1"}\n\n{"question":"Q2","answer":"A2"}\n')
            loaded = load_jsonl(p)
            assert len(loaded) == 2


# ─────────────────────────────────────────────────────────────────────────────
# validate_checkpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCheckpoint:
    def test_valid_checkpoint_with_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "adapter_config.json").write_text("{}")
            (p / "adapter_model.safetensors").write_text("fake")
            assert validate_checkpoint(p) is True

    def test_valid_checkpoint_with_bin(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "adapter_config.json").write_text("{}")
            (p / "adapter_model.bin").write_text("fake")
            assert validate_checkpoint(p) is True

    def test_missing_adapter_config_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "adapter_model.safetensors").write_text("fake")
            assert validate_checkpoint(p) is False

    def test_empty_directory_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            assert validate_checkpoint(d) is False


# ─────────────────────────────────────────────────────────────────────────────
# count_parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestCountParameters:
    def test_counts_trainable_and_total(self):
        """
        count_parameters() works on any object whose .parameters() returns
        objects with .numel() and .requires_grad.  We mock that interface
        so the test runs without torch or CUDA.
        """
        from unittest.mock import MagicMock

        def make_param(numel: int, requires_grad: bool):
            p = MagicMock()
            p.numel.return_value = numel
            p.requires_grad = requires_grad
            return p

        # Simulate: 220 frozen params + 105 trainable params
        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            make_param(220, requires_grad=False),
            make_param(105, requires_grad=True),
        ]

        trainable, total = count_parameters(mock_model)
        assert total     == 325
        assert trainable == 105
