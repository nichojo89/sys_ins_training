"""Synthetic data generation script for XML system prompt fine-tuning.

Generates synthetic banking customer service phone call conversations using
Claude Haiku, based on the XML system prompt defined in data/system_prompt.xml.
Output is saved in JSONL format for training, and a JSON test split is also
produced.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import anthropic

from src.config import (
    NUM_TRAINING_EXAMPLES,
    NUM_TEST_EXAMPLES,
    SYSTEM_PROMPT_PATH,
    DATA_DIR,
)

# Claude Haiku - cheap model for synthetic data generation.
# NOTE: claude-3-5-haiku-20241022 was retired (Feb 19, 2026); using its
# current replacement per Anthropic's model migration guide.
MODEL_NAME = "claude-haiku-4-5"

TRAINING_JSONL_PATH = DATA_DIR / "training.jsonl"
TEST_JSON_PATH = DATA_DIR / "test_conversations.json"

SCENARIO_TOPICS = [
    "checking account balance inquiry",
    "reporting a lost or stolen debit card",
    "disputing an unauthorized transaction",
    "asking about opening a new savings account",
    "requesting information about a loan application",
    "resetting online banking password",
    "asking about wire transfer fees",
    "reporting a suspected scam or phishing attempt",
    "asking about overdraft fees and how to avoid them",
    "requesting a copy of a bank statement",
    "asking about mobile check deposit",
    "closing an account",
    "asking about credit card rewards program",
    "inquiring about mortgage refinancing options",
    "setting up direct deposit",
    "asking about international transaction fees",
    "requesting a higher credit limit",
    "asking about CD (certificate of deposit) rates",
    "reporting an ATM that did not dispense cash",
    "asking about joint account setup",
]


def load_system_prompt() -> str:
    """Load the XML system prompt from data/system_prompt.xml."""
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _build_generation_prompt(system_prompt: str, topic: str) -> str:
    return f"""You are generating synthetic training data for fine-tuning a banking customer
service AI assistant named Morty. The assistant follows this XML system prompt:

{system_prompt}

Generate ONE realistic banking customer service phone call conversation about: "{topic}"

The conversation should have 3-5 turns total (a "turn" is one customer message followed
by one assistant response), covering things like: greeting, identity verification /
account lookup, addressing the customer's issue, and a resolution/closing.

Return ONLY a JSON object (no markdown, no array) with this exact schema:
{{
  "user": "the full customer side of the conversation, with turns separated by newlines, e.g. 'Customer: ...\\nMorty: ...\\nCustomer: ...'",
  "assistant": "Morty's ideal final response that resolves the conversation, in character as described in the system prompt"
}}

Make sure the conversation is realistic, follows the restrictions in the system prompt
(no passwords, proper identity verification, escalate when appropriate), and
reflects Morty's professional, empathetic, and helpful personality."""


def _parse_json_response(text: str) -> Dict:
    """Parse a JSON object out of a Claude response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # drop the first fence line (```json or ```)
        lines = lines[1:]
        # drop trailing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def generate_training_examples(num_examples: int) -> List[Dict[str, str]]:
    """Generate `num_examples` synthetic training examples using Claude Haiku.

    Generates examples one at a time for reliability. Each example is a dict with
    keys: "system", "user", "assistant".
    """
    system_prompt = load_system_prompt()
    client = anthropic.Anthropic()

    examples: List[Dict[str, str]] = []
    import sys

    print(f"Generating {num_examples} examples (1 per API call for reliability)...", flush=True, file=sys.stderr)

    for i in range(num_examples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i + 1}/{num_examples} examples generated...", flush=True, file=sys.stderr)

        topic = SCENARIO_TOPICS[i % len(SCENARIO_TOPICS)]
        prompt = _build_generation_prompt(system_prompt, topic)

        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            text = next(
                (block.text for block in response.content if block.type == "text"), ""
            )

            try:
                parsed = _parse_json_response(text)
                user_content = parsed.get("user", "")
                assistant_content = parsed.get("assistant", "")
            except (json.JSONDecodeError, KeyError, TypeError):
                # Fallback if parsing fails
                user_content = f"Customer: I have a question about {topic}."
                assistant_content = "I'd be happy to help. Could you tell me more about what you need?"

            examples.append({
                "system": system_prompt,
                "user": user_content,
                "assistant": assistant_content,
            })

        except Exception as e:
            print(f"Error on example {i + 1}: {type(e).__name__}. Using fallback.", flush=True, file=sys.stderr)
            examples.append({
                "system": system_prompt,
                "user": f"Customer: I need help with my banking ({topic}).",
                "assistant": "I'm here to help. What's your question?"
            })

    return examples


def save_to_jsonl(examples: List[Dict[str, str]], output_path: Path) -> None:
    """Save a list of examples to a JSONL file (one JSON object per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


def train_test_split(
    examples: List[Dict[str, str]], test_ratio: float = 0.15
) -> "tuple[List[Dict[str, str]], List[Dict[str, str]]]":
    """Split examples into (train, test) sets."""
    shuffled = examples[:]
    random.seed(42)
    random.shuffle(shuffled)

    num_test = int(len(shuffled) * test_ratio)
    test_set = shuffled[:num_test]
    train_set = shuffled[num_test:]
    return train_set, test_set


def generate_and_save_data() -> None:
    """Generate synthetic conversations and save train/test splits to disk."""
    total_examples = NUM_TRAINING_EXAMPLES + NUM_TEST_EXAMPLES
    print(f"Generating {total_examples} synthetic training examples using {MODEL_NAME}...")

    examples = generate_training_examples(total_examples)

    train_set, test_set = train_test_split(
        examples, test_ratio=NUM_TEST_EXAMPLES / total_examples
    )

    save_to_jsonl(train_set, TRAINING_JSONL_PATH)
    print(f"Saved {len(train_set)} training examples to {TRAINING_JSONL_PATH}")

    with open(TEST_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)
    print(f"Saved {len(test_set)} test examples to {TEST_JSON_PATH}")


if __name__ == "__main__":
    generate_and_save_data()
