"""Generate the Micro-BERT 2M + RL Colab notebook."""

import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "Micro_BERT_2M_RL.ipynb")


def code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def markdown_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


cells = []

# ===== Title =====
cells.append(markdown_cell(r"""
# Micro-BERT 2M + RL Training on Google Colab

> Train a **~2 million parameter** extractive QA model and fine-tune it with **Reinforcement Learning**.

This notebook covers:
1. **Supervised pre-training** on SQuAD 1.1
2. **RL fine-tuning** (REINFORCE or GRPO)
3. **Evaluation** on the SQuAD validation set

**Recommended runtime:** GPU (T4)
"""))

# ===== Setup =====
cells.append(code_cell(r"""
# @title 1. Clone repository & install dependencies
!git clone https://github.com/Bleibel/BERT_QUESTION_ANSWERING.git
%cd BERT_QUESTION_ANSWERING
!pip install -q transformers datasets accelerate torch flask python-dotenv
"""))

# ===== GPU Check =====
cells.append(code_cell(r"""
# @title 2. Check GPU availability
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")
"""))

# ===== Imports & Config =====
cells.append(markdown_cell(r"""
## Step 1: Define the ~2M Parameter Micro-BERT

Architecture:
- **Layers:** 2
- **Hidden size:** 64
- **Attention heads:** 2
- **Intermediate (FFN):** 128
- **Total parameters:** ~2,084,864
"""))

cells.append(code_cell(r"""
# @title 3. Imports and 2M model config
import os
import sys
import json

from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset

sys.path.insert(0, "/content/BERT_QUESTION_ANSWERING")
from src.micro_bert import get_micro_bert_2m_config, count_parameters
from train import prepare_train_features

# Define 2M config
config = get_micro_bert_2m_config()
model = BertForQuestionAnswering(config)

print("=" * 50)
print("Micro-BERT 2M Configuration")
print("=" * 50)
print(f"Hidden size      : {config.hidden_size}")
print(f"Layers           : {config.num_hidden_layers}")
print(f"Attention heads  : {config.num_attention_heads}")
print(f"Intermediate     : {config.intermediate_size}")
print(f"Max positions    : {config.max_position_embeddings}")
print(f"Total parameters : {count_parameters(model):,}")
print("=" * 50)
"""))

# ===== Load SQuAD =====
cells.append(markdown_cell(r"""
## Step 2: Load & Tokenize SQuAD 1.1

We download the full SQuAD training set from Hugging Face `datasets`.
For a quick test, you can limit `MAX_EXAMPLES` below.
"""))

cells.append(code_cell(r"""
# @title 4. Load SQuAD dataset
MAX_EXAMPLES = None  # Set to e.g. 5000 for a quick experiment

dataset = load_dataset("rajpurkar/squad", split="train")
if MAX_EXAMPLES:
    dataset = dataset.select(range(min(MAX_EXAMPLES, len(dataset))))

print(f"Training examples: {len(dataset)}")
"""))

# ===== Tokenize =====
cells.append(code_cell(r"""
# @title 5. Tokenize with sliding-window chunking
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenized_dataset = dataset.map(
    lambda x: prepare_train_features(x, tokenizer, max_length=384, doc_stride=128),
    batched=True,
    remove_columns=dataset.column_names,
)

print(f"Tokenized examples: {len(tokenized_dataset)}")
"""))

# ===== Supervised Training =====
cells.append(markdown_cell(r"""
## Step 3: Supervised Pre-training

Train the 2M model from scratch on SQuAD using the Hugging Face `Trainer` API.

| Setting | Value |
|---------|-------|
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | 3e-5 |
| Warmup | 10% |
| FP16 | Enabled on GPU |

**Expected time on Colab T4:** ~15-25 minutes for 3 epochs (full SQuAD)
"""))

cells.append(code_cell(r"""
# @title 6. Supervised training
import shutil

OUTPUT_DIR = "checkpoints/micro-bert-2m"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clean output dir to avoid resume conflicts
for item in os.listdir(OUTPUT_DIR):
    item_path = os.path.join(OUTPUT_DIR, item)
    if os.path.isfile(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=max(1, int(0.1 * len(tokenized_dataset) / 32)),
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator,
)

print("Starting supervised training...")
trainer.train()

# Save checkpoint
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata
meta = {
    "model_type": "micro-bert-2m-qa",
    "total_parameters": count_parameters(model),
    "config": config.to_dict(),
    "training": {
        "dataset": "squad",
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 3e-5,
    },
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSupervised checkpoint saved to: {OUTPUT_DIR}")
"""))

# ===== RL Section =====
cells.append(markdown_cell(r"""
## Step 4: RL Fine-tuning (Choose ONE)

We provide two RL options:

### Option A: REINFORCE (Simple, Fast)
Vanilla policy gradient with a running-average baseline.

### Option B: GRPO (DeepSeek-R1 Style)
Group Relative Policy Optimization — no critic network, uses group mean reward as baseline.

**Tip:** REINFORCE is more stable for tiny models. GRPO is more sophisticated but needs careful tuning.
"""))

# REINFORCE
cells.append(code_cell(r"""
# @title 7A. REINFORCE fine-tuning
!python train_rl.py \
    --checkpoint checkpoints/micro-bert-2m \
    --dataset squad \
    --output_dir checkpoints/micro-bert-2m-rl \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --device 0
"""))

# GRPO
cells.append(code_cell(r"""
# @title 7B. GRPO fine-tuning (alternative to REINFORCE)
# Uncomment and run this INSTEAD of the REINFORCE cell above.

# !python train_grpo.py \
#     --checkpoint checkpoints/micro-bert-2m \
#     --dataset squad \
#     --output_dir checkpoints/micro-bert-2m-grpo \
#     --epochs 5 \
#     --batch_size 8 \
#     --group_size 8 \
#     --learning_rate 1e-5 \
#     --device 0
"""))

# ===== Evaluation =====
cells.append(markdown_cell(r"""
## Step 5: Evaluation on SQuAD Validation

Evaluate the final RL checkpoint against the official SQuAD validation set.
`--max-examples` limits evaluation time on Colab.
"""))

cells.append(code_cell(r"""
# @title 8. Evaluate RL model
# Change MODEL_PATH if you used GRPO instead of REINFORCE
MODEL_PATH = "checkpoints/micro-bert-2m-rl"

!python run_evaluation.py \
    --dataset squad \
    --model {MODEL_PATH} \
    --max-examples 1000 \
    --device 0
"""))

# ===== Show results =====
cells.append(code_cell(r"""
# @title 9. Display results
import json

results_path = "results/eval_results.json"
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    cfg = results.get("config", {})
    metrics = results.get("metrics", {})
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Model           : {cfg.get('model', 'N/A')}")
    print(f"Parameters      : {cfg.get('num_parameters', 'N/A'):,}")
    print(f"Examples        : {cfg.get('num_examples', 'N/A')}")
    print(f"Inference time  : {cfg.get('total_inference_time', 0):.1f}s")
    print("-" * 50)
    print(f"Exact Match     : {metrics.get('exact_match', 0):.2f}%")
    print(f"F1 Score        : {metrics.get('f1', 0):.2f}%")
    print("=" * 50)
else:
    print(f"Results not found at {results_path}")
"""))

# ===== Save to Drive =====
cells.append(markdown_cell(r"""
## Step 6: Save Checkpoints to Google Drive (Optional)

Mount your Drive and copy the trained checkpoints for permanent storage.
"""))

cells.append(code_cell(r"""
# @title 10. Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create a destination folder
DRIVE_DIR = "/content/drive/MyDrive/micro-bert-2m-rl"
os.makedirs(DRIVE_DIR, exist_ok=True)

# Copy all checkpoints
!cp -r checkpoints/micro-bert-2m* {DRIVE_DIR}/
!cp -r results {DRIVE_DIR}/

print(f"Checkpoints saved to: {DRIVE_DIR}")
"""))

# ===== Quick test =====
cells.append(markdown_cell(r"""
## Bonus: Quick Interactive Test

Run inference on a custom passage + question.
"""))

cells.append(code_cell(r"""
# @title 11. Interactive inference
from src.model import BERTQA

# Load the RL checkpoint
qa = BERTQA(model_name="checkpoints/micro-bert-2m-rl", device=0 if torch.cuda.is_available() else -1)

passage = (
    "Berlin is the capital and largest city of Germany. "
    "Its 3.7 million inhabitants make it the EU's most populous city. "
    "The city is surrounded by the state of Brandenburg."
)

question = "What is the capital of Germany?"

result = qa.answer(question=question, context=passage)
print(f"Question : {question}")
print(f"Answer   : {result['answer']}")
print(f"Confidence: {result['score']:.4f}")
print(f"Position : {result['start']} - {result['end']}")
"""))

# ===== Notebook metadata =====
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "cells": cells,
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created: {NOTEBOOK_PATH}")
