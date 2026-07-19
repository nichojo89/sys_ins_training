#!/usr/bin/env python3
"""
Complete three-step XML System Prompt Fine-Tuning workflow for Google Colab.
Paste this into a Colab cell and run - it handles everything.

Steps:
1. Test base model (baseline)
2. Fine-tune with unsloth
3. Test fine-tuned model (evaluation)
"""

import os
import json
import torch
import subprocess
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("XML SYSTEM PROMPT FINE-TUNING: THREE-STEP WORKFLOW")
print("=" * 80)

# ============================================================================
# SETUP: Install dependencies and authenticate
# ============================================================================
print("\n[SETUP] Installing dependencies...")
subprocess.run(["pip", "install", "-q", "torch", "transformers", "datasets", "peft", "trl", "huggingface-hub"], check=False)
subprocess.run(["pip", "install", "-q", "git+https://github.com/unslothai/unsloth.git"], check=False)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from huggingface_hub import login

# Authenticate
print("[SETUP] Authenticate with Hugging Face (paste your token when prompted):")
hf_token = input("HF Token: ").strip()
login(token=hf_token)
print("✓ Authenticated\n")

# ============================================================================
# CREATE MINIMAL PROJECT STRUCTURE
# ============================================================================
print("[SETUP] Creating project structure...")

# Create directories
for d in ["data", "models", "output/results"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Create XML system prompt if it doesn't exist
if not Path("data/system_prompt.xml").exists():
    system_prompt_xml = """<assignment>
You are a customer service representative named Morty representing FinBank on a phone call with a user. Your goal is to assist them with their banking needs while maintaining professionalism and accuracy.
</assignment>

<character>
  <name>Morty</name>
  <company>FinBank</company>
  <primary_objective>
    Provide clear, accurate banking information that sounds natural and human-like. Handle customer inquiries with empathy and efficiency.
  </primary_objective>
  <tone>Professional but friendly, clear and concise</tone>
</character>

<restrictions>
  <rule>Never request passwords or full credit card numbers</rule>
  <rule>Always confirm account details before discussing sensitive information</rule>
  <rule>If the customer asks about a service you cannot handle, offer to transfer them to the right department</rule>
  <rule>Respond in a conversational manner, not robotic</rule>
</restrictions>

<job>
  <task name="Initial Greeting">
    <move name="Introduce yourself">
      <trigger>Call Starts</trigger>
      <action>Speak: "Hi! Thank you for calling FinBank. This is Morty. How can I help you today?"</action>
    </move>
  </task>
  <task name="Information Gathering">
    <move name="Request account number">
      <trigger>Customer states their need</trigger>
      <action>Speak: "I'd be happy to help. To look up your account, could I please get the first and last 4 digits of your account number?"</action>
    </move>
  </task>
  <task name="Resolution">
    <move name="Provide solution">
      <trigger>Account confirmed</trigger>
      <action>Provide relevant banking information or service</action>
    </move>
  </task>
</job>

<examples>
  <example description="Confirming account numbers">
    <turn speaker="user">My account is 1-2-3-4-5-6-7-8</turn>
    <turn speaker="model">Just to confirm, I have you down as account ending in 7-8. Is that correct?</turn>
  </example>
  <example description="Handling restrictions">
    <turn speaker="user">Can you tell me my password?</turn>
    <turn speaker="model">For security reasons, I can never access or share your password. If you've forgotten it, I can help you reset it securely. Would you like me to do that?</turn>
  </example>
</examples>"""

    with open("data/system_prompt.xml", "w") as f:
        f.write(system_prompt_xml)
    print("✓ Created system_prompt.xml")

# Load system prompt
with open("data/system_prompt.xml") as f:
    system_prompt = f.read()

# Create synthetic training data if it doesn't exist
if not Path("data/training.jsonl").exists():
    print("✓ Creating synthetic training data...")

    conversations = [
        ("Customer: Hi, I'm calling about my account.\nMorty: Hi there! Thank you for calling FinBank. This is Morty. How can I help you today?\nCustomer: I'd like to check my balance.\nMorty: I'd be happy to help. To look up your account, could I please get the first and last 4 digits of your account number?",
         "Of course! Once I have your account details, I can pull up your balance right away."),
        ("Customer: My debit card was lost yesterday.\nMorty: I'm sorry to hear that! I can help you with that. Let me confirm a few details. Can you provide the last 4 digits of your account number?\nCustomer: Sure, it's 4567.\nMorty: Thank you. Now, just to confirm, I have you down as account ending in 4567. Is that correct?",
         "Perfect! I'll get that card deactivated right away to protect your account."),
        ("Customer: I saw a charge on my account that I didn't make.\nMorty: I understand your concern. I'm here to help. Could you please confirm the last 4 digits of your account number?",
         "I can file a dispute for that transaction right away. We'll investigate within 10 business days."),
    ]

    with open("data/training.jsonl", "w") as f:
        for user_conv, assistant_resp in conversations * 150:  # Repeat 150x to get ~450 examples
            example = {"system": system_prompt, "user": user_conv, "assistant": assistant_resp}
            f.write(json.dumps(example) + "\n")
    print(f"✓ Created training.jsonl ({450} examples)")

# Load training data
dataset = load_dataset('json', data_files='data/training.jsonl')
train_dataset = dataset['train']
print(f"✓ Loaded {len(train_dataset)} training examples\n")

# ============================================================================
# STEP 1: BASELINE TEST - Test base model before fine-tuning
# ============================================================================
print("=" * 80)
print("STEP 1: BASELINE TEST - Testing Base Model (BEFORE fine-tuning)")
print("=" * 80)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"✓ Model loaded on {next(model.parameters()).device}")

# Test inputs
test_inputs = [
    "Hi, I want to check my account balance.",
    "My debit card was lost. What should I do?",
    "I saw a charge I didn't make."
]

baseline_results = []
print("\nRunning baseline inference...")

for i, user_input in enumerate(test_inputs, 1):
    print(f"\n  [{i}] {user_input}")

    prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    baseline_results.append({
        "input": user_input,
        "response": response,
        "model": "base"
    })
    print(f"      Response: {response[:100]}...")

print(f"\n✓ Baseline test complete ({len(baseline_results)} examples)")

# ============================================================================
# STEP 2: FINE-TUNING - Train with unsloth
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FINE-TUNING - Training with unsloth + LoRA")
print("=" * 80)

print("\nSetting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\nStarting training...")
training_args = SFTConfig(
    output_dir="./models/fine_tuned",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Smaller batch for Colab memory
    gradient_accumulation_steps=2,
    warmup_ratio=0.03,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    max_seq_length=4096,
    dataset_text_field="user",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save model
model.save_pretrained("./models/fine_tuned")
tokenizer.save_pretrained("./models/fine_tuned")
print("✓ Fine-tuned model saved to ./models/fine_tuned\n")

# ============================================================================
# STEP 3: EVALUATION TEST - Test fine-tuned model
# ============================================================================
print("=" * 80)
print("STEP 3: EVALUATION TEST - Testing Fine-Tuned Model (AFTER fine-tuning)")
print("=" * 80)

print("\nLoading fine-tuned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

fine_tuned_model = PeftModel.from_pretrained(base_model, "./models/fine_tuned")
fine_tuned_model = fine_tuned_model.merge_and_unload()
print("✓ Fine-tuned model loaded and merged")

evaluation_results = []
print("\nRunning evaluation inference...")

for i, user_input in enumerate(test_inputs, 1):
    print(f"\n  [{i}] {user_input}")

    prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(fine_tuned_model.device)

    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    evaluation_results.append({
        "input": user_input,
        "response": response,
        "model": "fine_tuned"
    })
    print(f"      Response: {response[:100]}...")

print(f"\n✓ Evaluation test complete ({len(evaluation_results)} examples)")

# ============================================================================
# COMPARISON: Before vs After
# ============================================================================
print("\n" + "=" * 80)
print("BEFORE vs AFTER COMPARISON")
print("=" * 80)

for i, (base, ft) in enumerate(zip(baseline_results, evaluation_results), 1):
    print(f"\n{'─'*80}")
    print(f"Example {i}: {base['input']}")
    print(f"{'─'*80}")
    print(f"\n[BASE MODEL]:\n{base['response'][:150]}...\n")
    print(f"[FINE-TUNED]:\n{ft['response'][:150]}...\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results = {
    "timestamp": datetime.now().isoformat(),
    "workflow": "XML System Prompt Fine-Tuning",
    "model": model_name,
    "training": {
        "epochs": 1,
        "examples": len(train_dataset),
        "batch_size": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32
    },
    "baseline_test": baseline_results,
    "evaluation_test": evaluation_results,
    "comparison": {
        "test_examples": len(test_inputs),
        "improvements": "See baseline_test vs evaluation_test for before/after responses"
    }
}

with open("output/results/workflow_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to output/results/workflow_results.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 WORKFLOW COMPLETE!")
print("=" * 80)
print(f"\n✅ Step 1: Baseline test ({len(baseline_results)} examples)")
print(f"✅ Step 2: Fine-tuning ({len(train_dataset)} training examples)")
print(f"✅ Step 3: Evaluation test ({len(evaluation_results)} examples)")
print(f"\n📊 Results saved to: output/results/workflow_results.json")
print(f"💾 Model checkpoint saved to: ./models/fine_tuned")
print("\n✓ Ready to download results and analyze improvements!")
