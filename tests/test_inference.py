"""Unit tests for the inference script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import run_inference, load_finetuned_model


def test_run_inference_returns_string():
    response = run_inference("base", "Hi, I want to check my balance", "You are Morty...")
    assert isinstance(response, str)
    assert len(response) > 0


def test_load_finetuned_model():
    model, tokenizer = load_finetuned_model()
    assert model is not None
    assert tokenizer is not None
