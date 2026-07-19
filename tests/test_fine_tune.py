"""Unit tests for the fine-tuning setup script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fine_tune import load_model_and_tokenizer


def test_load_model_and_tokenizer():
    model, tokenizer = load_model_and_tokenizer()
    assert model is not None
    assert tokenizer is not None
    assert hasattr(model, "forward")
    assert hasattr(tokenizer, "encode")
