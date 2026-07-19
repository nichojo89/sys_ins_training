# XML System Prompt Fine-Tuning

A demonstration project for fine-tuning Llama 2 7B to follow XML-based system prompts for banking customer service scenarios.

## Setup Instructions

Follow these steps to set up the project:

1. **Clone the repository and navigate to the project directory**
   ```bash
   cd sys_ins_training
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your Hugging Face credentials**
   ```bash
   huggingface-cli login
   ```

5. **Verify the setup**
   ```bash
   python -c "from src.config import *; print('Setup successful!')"
   ```

## Project Structure

### Directory Layout

- **data/** - Training data and system prompts
  - `system_prompt.xml` - XML template for banking customer service scenarios
  - `training_data.json` - Generated training dataset
  - `test_data.json` - Test dataset for evaluation
  - `validation_data.json` - Validation dataset for hyperparameter tuning

- **src/** - Source code and configuration
  - `__init__.py` - Package initialization
  - `config.py` - Configuration settings and hyperparameters
  - Additional modules for data processing and training

- **models/** - Model checkpoints and fine-tuned models
  - Stores fine-tuned model checkpoints during training
  - Final fine-tuned model after completion

- **notebooks/** - Jupyter notebooks for development and experimentation
  - Data generation and exploration
  - Model training and fine-tuning
  - Evaluation and inference

## Evaluation

After running the full pipeline (data generation → fine-tuning → inference), evaluate the fine-tuned model using:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

The notebook guides manual comparison of base vs fine-tuned responses on:
1. XML structure adherence (task routing, restrictions)
2. Example consistency (tone, format, behavioral patterns)

See `EVALUATION_RESULTS.md` for detailed findings and example comparisons.

## Key Features

- **XML-based system prompts** for structured prompt management
- **LoRA fine-tuning** for efficient parameter updates
- **Unsloth integration** for fast training on consumer GPUs
- **Banking customer service scenarios** with 400 training examples
- **Comprehensive configuration** in `src/config.py`

## Expected Results

Upon successful completion of fine-tuning:

- Model achieves improved performance on XML-based system prompt instructions
- Customer service responses follow the specified XML structure
- Training completes in 2-3 hours on GPU (varies by hardware)
- Fine-tuned model demonstrates better adherence to role-playing instructions
- Validation metrics show improvement over base model

## Configuration

All training parameters can be modified in `src/config.py`:

- Model selection: `BASE_MODEL_NAME`
- Training dataset size: `NUM_TRAINING_EXAMPLES`, `NUM_TEST_EXAMPLES`
- Model parameters: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`
- Training hyperparameters: `LEARNING_RATE`, `NUM_EPOCHS`, `TRAIN_BATCH_SIZE`

## License

This project is provided for educational purposes.
