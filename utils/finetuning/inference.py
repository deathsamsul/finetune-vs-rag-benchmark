"""
utils/finetuning/inference.py — Compatibility alias for FineTunedInference.

Re-exports FineTunedInference from utils.fine_tune_inference to avoid duplicate code.
"""

from utils.fine_tune_inference import FineTunedInference

__all__ = ["FineTunedInference"]
