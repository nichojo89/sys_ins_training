# XML-Based System Prompt Fine-Tuning Design

**Date:** 2026-07-18  
**Project:** Training a model to follow custom XML-based prompt syntax for system instructions  
**Scope:** Demonstration/research project to evaluate behavior before and after instruction fine-tuning  
**Target Model:** Llama 3.1 8B with unsloth  

---

## Overview

This project demonstrates how fine-tuning a language model on examples of structured XML system prompts improves the model's ability to follow complex, role-based conversational instructions. The evaluation compares model behavior before and after fine-tuning on a banking customer service phone call scenario.

## Use Case

Train and evaluate a language model to:
- Parse and understand XML-structured system instructions that define character, objectives, restrictions, and task workflows
- Follow task routing and conversational steps defined in XML
- Adhere to behavioral examples provided in XML format
- Maintain consistency across multi-turn conversations with a large context window

## Design Decisions

### Base Model: Llama 3.1 8B

- **8K context window** required for long conversational evaluation
- **Instruction-following baseline** strong enough to measure improvement
- **Unsloth compatibility** fully optimized kernels available
- **Trade-off:** 8B is larger than 1B but more convincing for demo purposes; small enough for quick iteration

### Fine-Tuning Approach: XML-as-System-Prompt

**Structure:** Each training example follows standard instruction-format:

```
[System Prompt]
<assignment>...</assignment>
<character>...</character>
<restrictions>...</restrictions>
<job>...</job>
<examples>...</examples>

[User Input]
"<conversational turn from simulated phone call>"

[Expected Output]
"<model response following XML structure and examples>"
```

**Why this approach:**
- Reusable across different XML specifications
- Natural prompt engineering paradigm (no custom tokenization)
- Clear before/after comparison (same XML spec used in both)
- Easy to debug and iterate

### Synthetic Data Generation: 300-500 Examples

**Scale rationale:**
- Too small (<100): Risk of overfitting, weak improvement signal
- **Sweet spot (300-500):** Clear fine-tuning effect visible in manual evaluation; tractable to generate and evaluate
- Too large (1000+): Overkill for demo, tedious manual evaluation, slower generation

**Generation method:**
- Create conversations in a banking customer service phone call scenario
- Each multi-turn conversation becomes multiple training examples (one per turn)
- Cover scenarios:
  - Initial greeting and task introduction
  - Information gathering (account numbers, customer needs)
  - Task execution following the job workflow
  - Edge cases (customer confusion, restriction violations)

**Format:** `.jsonl` (one JSON object per line, compatible with unsloth)

```json
{
  "system": "<assignment>...</assignment>\n<character>...</character>\n...",
  "user": "user conversational input",
  "assistant": "expected model response"
}
```

### Fine-Tuning Configuration

- **Framework:** unsloth with LoRA
- **Epochs:** 1 (prevents overfitting, appropriate for demo scale)
- **Learning rate:** unsloth defaults
- **Batch size:** Determined by GPU memory (unsloth handles optimization)
- **Output:** Merged fine-tuned model checkpoint

### Evaluation Framework: Manual Before/After Comparison

**Pre-fine-tuning baseline:**
1. Select 50-100 held-out test conversations (not in training set)
2. Run through base Llama 3.1 8B with the XML system prompt
3. Record responses for manual review

**Post-fine-tuning evaluation:**
1. Run same test conversations through fine-tuned model
2. Manual comparison on two dimensions:
   - **XML structure adherence:** Does the model follow task routing, respect restrictions, execute moves in order?
   - **Example consistency:** Do responses match the tone, format, and behavioral patterns shown in examples?
3. Document 5-10 side-by-side before/after examples showing clearest improvements

## Project Deliverables

1. **Training dataset:** Synthetic 300-500 example conversations in `.jsonl` format
2. **Fine-tuned model:** Merged checkpoint ready for inference
3. **Evaluation report:** Before/after examples with manual assessment
4. **Inference script:** Python script to run conversations with XML system prompt (base vs fine-tuned)

## Success Criteria

- Model demonstrates measurable improvement in XML structure adherence post fine-tuning
- Model responses show improved consistency with provided examples
- Evaluation clearly shows the value of fine-tuning for instruction-following on structured prompts

## Technical Stack

- **Model serving:** unsloth (fine-tuning)
- **Data format:** `.jsonl`
- **Evaluation:** Manual review + optional automated metrics (pattern matching)
- **Model framework:** Hugging Face transformers
- **Data generation:** Claude Haiku (cheap, repetitive task)

## Out of Scope

- Production deployment
- Automated evaluation metrics (manual only)
- Multi-model comparison (Llama 3.1 8B only)
- Custom tokenization or special XML tokens
