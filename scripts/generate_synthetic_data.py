#!/usr/bin/env python3
"""Generate synthetic banking conversation data without API calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_TRAINING_EXAMPLES,
    NUM_TEST_EXAMPLES,
    DATA_DIR,
    SYSTEM_PROMPT_PATH,
)

TRAINING_JSONL_PATH = DATA_DIR / "training.jsonl"
TEST_JSON_PATH = DATA_DIR / "test_conversations.json"

def load_system_prompt():
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read()

# Synthetic conversation templates
CONVERSATIONS = [
    {
        "user": "Customer: Hi, I'm calling about my account.\nMorty: Hi there! Thank you for calling FinBank. This is Morty. How can I help you today?\nCustomer: I'd like to check my balance.\nMorty: I'd be happy to help. To look up your account, could I please get the first and last 4 digits of your account number?",
        "assistant": "Of course! Once I have your account details, I can pull up your balance right away."
    },
    {
        "user": "Customer: My debit card was lost yesterday.\nMorty: I'm sorry to hear that! I can help you with that. Let me confirm a few details. Can you provide the last 4 digits of your account number?\nCustomer: Sure, it's 4567.\nMorty: Thank you. Now, just to confirm, I have you down as account ending in 4567. Is that correct?",
        "assistant": "Perfect! I'll get that card deactivated right away to protect your account. We can have a replacement card sent to you within 3-5 business days."
    },
    {
        "user": "Customer: I saw a charge on my account that I didn't make.\nMorty: I understand your concern. I'm here to help. Could you please confirm the last 4 digits of your account number?\nCustomer: It's 8901.\nMorty: Thank you. Now, when did you first notice this unauthorized charge?",
        "assistant": "I can file a dispute for that transaction right away. We'll investigate and get back to you within 10 business days with the results."
    },
    {
        "user": "Customer: Can you reset my online banking password?\nMorty: Of course, I can help with that. For security reasons, I can never access or share your password. What I can do is help you reset it securely. Do you still have access to your email?",
        "assistant": "I'll send a password reset link to your email on file. You'll be able to create a new password from there. Check your inbox and spam folder."
    },
    {
        "user": "Customer: What are the wire transfer fees?\nMorty: Great question! Let me pull up that information for you. Wire transfer fees can vary depending on whether it's domestic or international. Are you looking to send a domestic or international wire?",
        "assistant": "Domestic wires are $15, and international wires are $45. Both typically process within 1-2 business days."
    },
    {
        "user": "Customer: I want to open a savings account.\nMorty: Excellent! We have some great savings products. I can definitely help you with that. To get started, I'll need to confirm your identity. Could I please get your date of birth?",
        "assistant": "Our savings accounts have no monthly fees and competitive interest rates. I can get you set up right now and explain all the features."
    },
    {
        "user": "Customer: I got a suspicious email claiming to be from your bank.\nMorty: Thank you for reporting that! That sounds like a phishing attempt. For security reasons, we would never ask for passwords or full account numbers via email.\nCustomer: What should I do?\nMorty: Please delete the email and never click on links from unsolicited emails.",
        "assistant": "You did the right thing by calling us. If you get more suspicious emails, please forward them to our fraud department. Your account is safe."
    },
    {
        "user": "Customer: I'm having trouble with the mobile app.\nMorty: I'm sorry you're experiencing issues. Let me help troubleshoot. What exactly is happening when you try to use the app?\nCustomer: It keeps saying 'Connection Error'.\nMorty: Have you tried restarting the app and your phone?",
        "assistant": "If restarting doesn't work, try uninstalling and reinstalling the app. If you continue to have issues, I can escalate this to our technical support team."
    },
    {
        "user": "Customer: Can I set up direct deposit?\nMorty: Absolutely! Direct deposit is easy to set up. You'll just need to provide your employer with your banking information. Do you have your account number and routing number handy?",
        "assistant": "I can read those to you now, or you can find them on your checks. Direct deposit typically starts within 1-2 pay cycles."
    },
    {
        "user": "Customer: What are overdraft fees?\nMorty: Good question. Our overdraft fee is $35 per transaction that would overdraw your account. We also offer overdraft protection if you'd like.\nCustomer: How does that work?\nMorty: With overdraft protection, we can link your savings account or a credit line to cover overdrafts.",
        "assistant": "That way, instead of an overdraft fee, you'd just transfer funds between accounts. It's a great way to avoid fees and protect your account."
    },
]

def generate_synthetic_data(num_examples: int):
    """Generate synthetic data by cycling through conversation templates."""
    system_prompt = load_system_prompt()
    examples = []

    for i in range(num_examples):
        conv = CONVERSATIONS[i % len(CONVERSATIONS)]
        examples.append({
            "system": system_prompt,
            "user": conv["user"],
            "assistant": conv["assistant"],
        })

    return examples

def save_to_jsonl(examples, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

def train_test_split(examples, test_ratio=0.15):
    import random
    shuffled = examples[:]
    random.seed(42)
    random.shuffle(shuffled)

    num_test = int(len(shuffled) * test_ratio)
    test_set = shuffled[:num_test]
    train_set = shuffled[num_test:]
    return train_set, test_set

if __name__ == "__main__":
    total = NUM_TRAINING_EXAMPLES + NUM_TEST_EXAMPLES
    print(f"Generating {total} synthetic examples...")

    examples = generate_synthetic_data(total)
    print(f"✓ Generated {len(examples)} examples")

    train_set, test_set = train_test_split(examples, NUM_TEST_EXAMPLES / total)
    print(f"✓ Split: {len(train_set)} train, {len(test_set)} test")

    save_to_jsonl(train_set, TRAINING_JSONL_PATH)
    print(f"✓ Saved {len(train_set)} to {TRAINING_JSONL_PATH}")

    import json
    with open(TEST_JSON_PATH, 'w') as f:
        json.dump(test_set, f, indent=2)
    print(f"✓ Saved {len(test_set)} to {TEST_JSON_PATH}")

    print(f"\n✓ Data generation complete!")
