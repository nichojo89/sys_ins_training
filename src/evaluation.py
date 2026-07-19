"""Evaluation framework for comparing base and fine-tuned model responses.

This script generates base and fine-tuned model responses for every example in
the test set, and saves the results to JSON for manual review in
notebooks/evaluation.ipynb.

Note: running evaluate_responses()/run_full_evaluation() requires a GPU
environment with torch, transformers, and peft installed (same requirements
as fine-tuning). It will fail on a CPU-only machine, but the logic is correct
and intended to be run on the GPU machine after fine-tuning completes.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, RESULTS_DIR
from src.inference import run_inference, load_system_prompt

# The test set produced by data_generation.py / used throughout the project.
TEST_DATA_PATH = DATA_DIR / "test_conversations.json"


def load_test_data() -> List[Dict]:
    """Load test examples from data/test_conversations.json."""
    with open(TEST_DATA_PATH, "r") as f:
        return json.load(f)


def evaluate_responses(test_examples: List[Dict]) -> List[Dict]:
    """Generate base and fine-tuned model responses for each test example.

    Args:
        test_examples: list of dicts, each with at least "user" and
            "assistant" keys (and optionally "system" for a per-example
            system prompt).

    Returns:
        List of dicts with keys: example_id, user_input, expected_response,
        base_response, finetuned_response, notes.
    """
    default_system_prompt = load_system_prompt()

    results = []
    for idx, example in enumerate(test_examples):
        system_prompt = example.get("system", default_system_prompt)
        user_input = example["user"]
        expected_response = example.get("assistant", "")

        base_response = run_inference("base", user_input, system_prompt)
        finetuned_response = run_inference("finetuned", user_input, system_prompt)

        results.append(
            {
                "example_id": idx,
                "user_input": user_input,
                "expected_response": expected_response,
                "base_response": base_response,
                "finetuned_response": finetuned_response,
                "notes": "",
            }
        )

    return results


def save_evaluation_results(results: List[Dict], output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def run_full_evaluation() -> List[Dict]:
    """Load test data, run evaluation, save results, and return them."""
    test_examples = load_test_data()
    results = evaluate_responses(test_examples)

    output_path = RESULTS_DIR / "evaluation_results.json"
    save_evaluation_results(results, output_path)

    print(f"Evaluated {len(results)} examples.")
    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_full_evaluation()
