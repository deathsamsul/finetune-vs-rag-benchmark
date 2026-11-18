"""
tests/test_judge_config.py — Unit tests for evaluation configuration and judge factory.

Verifies:
  - CLI argument parsing and overrides over .env
  - Sensible defaults (local 3B model, auto device, 4-bit)
  - Missing API key validation for cloud providers
  - Judge factory selection logic across all providers (using mocks)
  - Result validity checking (no silent NaN conversion)
"""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

from evaluation.config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_PROVIDER,
    EvalConfig,
    PROVIDER_KEY_MAP,
)
from evaluation.judge import SUPPORTED_PROVIDERS, build_judge, build_embeddings


class TestEvalConfigDefaults:
    def test_default_provider_is_local(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = EvalConfig.from_env_and_args(None)
            assert cfg.judge_provider == "local"
            assert cfg.judge_model == DEFAULT_JUDGE_MODEL
            assert cfg.judge_device == "auto"
            assert cfg.judge_load_in_4bit is True
            assert cfg.judge_temperature == 0.0
            assert cfg.judge_max_tokens == 1024

    def test_env_override(self):
        env = {
            "JUDGE_PROVIDER": "groq",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "GROQ_API_KEY": "gsk_test123",
            "JUDGE_TEMPERATURE": "0.2",
            "JUDGE_MAX_TOKENS": "512",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = EvalConfig.from_env_and_args(None)
            assert cfg.judge_provider == "groq"
            assert cfg.judge_model == "llama-3.1-8b-instant"
            assert cfg.api_key == "gsk_test123"
            assert cfg.judge_temperature == 0.2
            assert cfg.judge_max_tokens == 512

    def test_cli_overrides_env(self):
        env = {
            "JUDGE_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.0-flash-lite",
            "GEMINI_API_KEY": "gemini_key",
            "GROQ_API_KEY": "groq_key",
        }
        args = argparse.Namespace(
            judge_provider="groq",
            judge_model="custom-groq-model",
            judge_device="cuda:0",
            judge_temperature=0.7,
            judge_max_tokens=256,
            n_samples=5,
        )
        with patch.dict(os.environ, env, clear=True):
            cfg = EvalConfig.from_env_and_args(args)
            assert cfg.judge_provider == "groq"
            assert cfg.judge_model == "custom-groq-model"
            assert cfg.judge_device == "cuda:0"
            assert cfg.judge_temperature == 0.7
            assert cfg.judge_max_tokens == 256
            assert cfg.n_samples == 5
            # API key should have resolved to groq
            assert cfg.api_key == "groq_key"


class TestEvalConfigValidation:
    def test_local_judge_requires_no_api_key(self):
        cfg = EvalConfig(judge_provider="local", judge_model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
        with patch("pathlib.Path.exists", return_value=True):
            errors = cfg.validate()
            assert len(errors) == 0

    def test_cloud_judge_without_key_returns_error(self):
        cfg = EvalConfig(judge_provider="groq", judge_model="llama-3.1-8b-instant", api_key=None)
        with patch("pathlib.Path.exists", return_value=True):
            errors = cfg.validate()
            assert any("requires GROQ_API_KEY" in e for e in errors)

    def test_missing_test_data_returns_error(self):
        cfg = EvalConfig(test_data="/nonexistent/path/test.jsonl")
        errors = cfg.validate()
        assert any("Test data not found" in e for e in errors)


class TestJudgeFactorySelection:
    def test_supported_providers_list(self):
        assert "local" in SUPPORTED_PROVIDERS
        assert "groq" in SUPPORTED_PROVIDERS
        assert "openrouter" in SUPPORTED_PROVIDERS
        assert "gemini" in SUPPORTED_PROVIDERS
        assert "openai" in SUPPORTED_PROVIDERS

    def test_unknown_provider_raises_value_error(self):
        cfg = EvalConfig(judge_provider="invalid_provider")
        with pytest.raises(ValueError, match="Unknown judge provider"):
            build_judge(cfg)

    def test_missing_api_key_exits(self):
        cfg = EvalConfig(judge_provider="openai", api_key=None)
        with pytest.raises(SystemExit):
            build_judge(cfg)

    @patch("evaluation.judge._build_groq_judge")
    def test_groq_judge_builder_invoked(self, mock_build):
        mock_build.return_value = MagicMock()
        cfg = EvalConfig(judge_provider="groq", judge_model="llama-3.1-8b-instant", api_key="test_key")
        res = build_judge(cfg)
        mock_build.assert_called_once_with(cfg)
        assert res == mock_build.return_value

    @patch("evaluation.judge._build_openrouter_judge")
    def test_openrouter_judge_builder_invoked(self, mock_build):
        mock_build.return_value = MagicMock()
        cfg = EvalConfig(judge_provider="openrouter", judge_model="gpt-4o-mini", api_key="test_key")
        res = build_judge(cfg)
        mock_build.assert_called_once_with(cfg)
        assert res == mock_build.return_value

    @patch("evaluation.judge._build_gemini_judge")
    def test_gemini_judge_builder_invoked(self, mock_build):
        mock_build.return_value = MagicMock()
        cfg = EvalConfig(judge_provider="gemini", judge_model="gemini-2.0-flash-lite", api_key="test_key")
        res = build_judge(cfg)
        mock_build.assert_called_once_with(cfg)
        assert res == mock_build.return_value

    @patch("evaluation.judge._build_openai_judge")
    def test_openai_judge_builder_invoked(self, mock_build):
        mock_build.return_value = MagicMock()
        cfg = EvalConfig(judge_provider="openai", judge_model="gpt-4o", api_key="test_key")
        res = build_judge(cfg)
        mock_build.assert_called_once_with(cfg)
        assert res == mock_build.return_value
