"""Fine-tuning script using unsloth and LoRA for the banking XML system prompt model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset

from src import config


def load_model_and_tokenizer():
    """Load the base Llama model and tokenizer with 4-bit quantization via unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.BASE_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_lora(model):
    """Configure LoRA adapters on the given model for efficient fine-tuning."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        task_type=config.LORA_TASK_TYPE,
        use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        random_state=config.SEED,
    )
    return model


def load_training_data():
    """Load the training dataset from the training data JSONL file."""
    dataset = load_dataset("json", data_files=str(config.TRAINING_DATA_PATH), split="train")
    return dataset


def train_model():
    """Load model, apply LoRA, load data, train, and save the fine-tuned model."""
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_model_and_tokenizer()
    model = setup_lora(model)
    dataset = load_training_data()

    sft_config = SFTConfig(
        output_dir=str(config.FINE_TUNED_MODEL_PATH),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=config.WARMUP_RATIO,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_text_field="user",
        packing=False,
        logging_steps=config.LOG_STEPS,
        save_steps=config.SAVE_STEPS,
        seed=config.SEED,
        fp16=config.USE_FP16,
        bf16=config.USE_BF16,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(str(config.FINE_TUNED_MODEL_PATH))
    tokenizer.save_pretrained(str(config.FINE_TUNED_MODEL_PATH))

    return trainer


if __name__ == "__main__":
    train_model()
