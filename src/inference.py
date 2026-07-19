"""Inference script for comparing base and fine-tuned models on the XML system prompt task."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src import config


def load_base_model():
    """Load the base pretrained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    return model, tokenizer


def load_finetuned_model():
    """Load the base model and apply the fine-tuned LoRA weights, merged for inference."""
    model, tokenizer = load_base_model()

    model = PeftModel.from_pretrained(model, str(config.FINE_TUNED_MODEL_PATH))
    model = model.merge_and_unload()

    return model, tokenizer


def load_system_prompt():
    """Load the XML system prompt from the data directory."""
    with open(config.SYSTEM_PROMPT_PATH, "r") as f:
        return f.read()


def run_inference(model_type, user_input, system_prompt):
    """Run inference on either the 'base' or 'finetuned' model and return the response string."""
    if model_type == "base":
        model, tokenizer = load_base_model()
    elif model_type == "finetuned":
        model, tokenizer = load_finetuned_model()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("Assistant:")[-1].strip()

    return response


def compare_models(test_input):
    """Compare base and fine-tuned model responses for the same test input."""
    system_prompt = load_system_prompt()

    base_response = run_inference("base", test_input, system_prompt)
    finetuned_response = run_inference("finetuned", test_input, system_prompt)

    print("=== Base Model Response ===")
    print(base_response)
    print()
    print("=== Fine-tuned Model Response ===")
    print(finetuned_response)

    return {
        "base": base_response,
        "finetuned": finetuned_response,
    }


if __name__ == "__main__":
    test_inputs = [
        "Hi, I want to check my balance",
        "I need to report a lost card",
        "Can you help me transfer money to my savings account?",
    ]

    for test_input in test_inputs:
        print(f"--- Test input: {test_input} ---")
        compare_models(test_input)
        print()
