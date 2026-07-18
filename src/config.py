"""Configuration for XML system prompt fine-tuning project."""

from pathlib import Path
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MODEL_TYPE = "llama"

# Data configuration
NUM_TRAINING_EXAMPLES = 400
NUM_TEST_EXAMPLES = 75
NUM_VALIDATION_EXAMPLES = 50
TOTAL_EXAMPLES = NUM_TRAINING_EXAMPLES + NUM_TEST_EXAMPLES + NUM_VALIDATION_EXAMPLES

# Text generation parameters
MAX_SEQ_LENGTH = 4096
PADDING_SIDE = "right"
TRUNCATION_SIDE = "right"

# Training hyperparameters
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CHECKPOINTING = True

# Batch sizes
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8

# Training duration
NUM_EPOCHS = 3
MAX_STEPS = -1
SAVE_STEPS = 50
EVAL_STEPS = 50
LOG_STEPS = 10

# LoRA configuration
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# Unsloth configuration
USE_UNSLOTH = True
UNSLOTH_MAX_SEQ_LENGTH = MAX_SEQ_LENGTH

# File paths
SYSTEM_PROMPT_PATH = DATA_DIR / "system_prompt.xml"
TRAINING_DATA_PATH = DATA_DIR / "training_data.json"
TEST_DATA_PATH = DATA_DIR / "test_data.json"
VALIDATION_DATA_PATH = DATA_DIR / "validation_data.json"

# Output paths
FINE_TUNED_MODEL_PATH = MODELS_DIR / "fine_tuned_model"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"

# Device configuration
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
USE_FP16 = True
USE_BF16 = False

# Miscellaneous
SEED = 42
REPORT_TO = ["tensorboard"]
LOGGING_DIR = str(LOGS_DIR)
DATALOADER_PIN_MEMORY = True
DATALOADER_NUM_WORKERS = 4
REMOVE_UNUSED_COLUMNS = False
