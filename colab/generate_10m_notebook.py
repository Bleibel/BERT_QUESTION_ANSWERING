"""Generate the Micro-BERT 10M Supervised Training Colab notebook."""

import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "Micro_BERT_10M_Supervised.ipynb")


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
# Micro-BERT 10M — Supervised Training from Scratch

> Train the **~10 million parameter** Micro-BERT on SQuAD 1.1 using Google Colab's free GPU.

**What's inside:**
- Full supervised fine-tuning (10 epochs, no RL)
- Evaluation on SQuAD validation set
- Save checkpoint to Google Drive
- Interactive inference test

**Expected time on T4 GPU:** ~25–35 minutes
"""))

# ===== Setup =====
cells.append(code_cell(r"""
# @title 1. Clone repository & install dependencies
!git clone https://github.com/Bleibel/BERT_QUESTION_ANSWERING.git
%cd BERT_QUESTION_ANSWERING
!pip install -q transformers datasets accelerate torch
"""))

# ===== GPU Check =====
cells.append(code_cell(r"""
# @title 2. Check GPU
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected. Training will be extremely slow on CPU.")
    print("Go to Runtime → Change runtime type → GPU")
"""))

# ===== Imports =====
cells.append(code_cell(r"""
# @title 3. Imports
import os
import sys
import json
import shutil

from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset

sys.path.insert(0, "/content/BERT_QUESTION_ANSWERING")
from src.micro_bert import get_micro_bert_config, count_parameters
from train import prepare_train_features
"""))

# ===== Config =====
cells.append(markdown_cell(r"""
## Step 1: Initialize the 10M Parameter Micro-BERT

Architecture:
- **Layers:** 4
- **Hidden size:** 256
- **Attention heads:** 4
- **Intermediate (FFN):** 512
- **Total parameters:** ~10,120,450
"""))

cells.append(code_cell(r"""
# @title 4. Initialize model & tokenizer
config = get_micro_bert_config()
model = BertForQuestionAnswering(config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("=" * 55)
print("Micro-BERT 10M Configuration")
print("=" * 55)
print(f"Hidden size      : {config.hidden_size}")
print(f"Layers           : {config.num_hidden_layers}")
print(f"Attention heads  : {config.num_attention_heads}")
print(f"Intermediate     : {config.intermediate_size}")
print(f"Max positions    : {config.max_position_embeddings}")
print(f"Total parameters : {count_parameters(model):,}")
print("=" * 55)
"""))

# ===== Dataset =====
cells.append(markdown_cell(r"""
## Step 2: Load SQuAD 1.1

We use the full **SQuAD 1.1** training set (~87K examples).

If you want a quick test first, set `MAX_EXAMPLES` below to e.g. `5000`.
"""))

cells.append(code_cell(r"""
# @title 5. Load dataset
MAX_EXAMPLES = None  # Set to 5000 for a quick 5-minute test run

print("Downloading SQuAD 1.1...")
dataset = load_dataset("rajpurkar/squad", split="train")

if MAX_EXAMPLES:
    dataset = dataset.select(range(min(MAX_EXAMPLES, len(dataset))))

print(f"Training examples: {len(dataset)}")
"""))

# ===== Tokenize =====
cells.append(code_cell(r"""
# @title 6. Tokenize with sliding-window chunking
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    lambda x: prepare_train_features(x, tokenizer, max_length=384, doc_stride=128),
    batched=True,
    remove_columns=dataset.column_names,
)
print(f"Tokenized examples: {len(tokenized_dataset)}")
"""))

# ===== Training =====
cells.append(markdown_cell(r"""
## Step 3: Supervised Training

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | 3e-5 |
| Warmup | 10% |
| Weight decay | 0.01 |
| FP16 | Enabled |

These are the project defaults. Adjust if needed.
"""))

cells.append(code_cell(r"""
# @title 7. Train
OUTPUT_DIR = "checkpoints/micro-bert-10m"
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
    num_train_epochs=10,
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

print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Metadata
meta = {
    "model_type": "micro-bert-10m-qa",
    "total_parameters": count_parameters(model),
    "config": config.to_dict(),
    "training": {
        "dataset": "squad",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 3e-5,
    },
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nCheckpoint saved to: {OUTPUT_DIR}")
"""))

# ===== Evaluation =====
cells.append(markdown_cell(r"""
## Step 4: Evaluate on SQuAD Validation

Test the model on 1,000 validation examples.
"""))

cells.append(code_cell(r"""
# @title 8. Evaluate
!python run_evaluation.py \
    --dataset squad \
    --model checkpoints/micro-bert-10m \
    --max-examples 1000 \
    --device 0
"""))

# ===== Show results =====
cells.append(code_cell(r"""
# @title 9. Display evaluation results
import json

results_path = "results/eval_results.json"
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    cfg = results.get("config", {})
    metrics = results.get("metrics", {})
    print("=" * 50)
    print("SQuAD Validation Results (10M Model)")
    print("=" * 50)
    print(f"Model           : {cfg.get('model', 'N/A')}")
    print(f"Parameters      : {cfg.get('num_parameters', 'N/A'):,}")
    print(f"Examples        : {cfg.get('num_examples', 'N/A')}")
    print("-" * 50)
    print(f"Exact Match     : {metrics.get('exact_match', 0):.2f}%")
    print(f"F1 Score        : {metrics.get('f1', 0):.2f}%")
    print("=" * 50)
else:
    print(f"Results not found at {results_path}")
"""))

# ===== Save to Drive =====
cells.append(markdown_cell(r"""
## Step 5: Save to Google Drive (Optional)

Keep your checkpoint permanently.
"""))

cells.append(code_cell(r"""
# @title 10. Save checkpoint to Drive
from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = "/content/drive/MyDrive"
SRC = "checkpoints/micro-bert-10m"
DST = f"{DRIVE_BASE}/micro-bert-10m"

if os.path.exists(SRC):
    !mkdir -p {DST}
    !cp -r {SRC}/* {DST}/
    print(f"Saved to: {DST}")
else:
    print(f"Checkpoint not found at {SRC}")
"""))

# ===== Interactive Test =====
cells.append(markdown_cell(r"""
## Bonus: Interactive Test

Run inference on any passage + question.
"""))

cells.append(code_cell(r"""
# @title 11. Test your model
from src.model import BERTQA

qa = BERTQA(model_name="checkpoints/micro-bert-10m", device=0 if torch.cuda.is_available() else -1)

passage = (
    "The University of Notre Dame began late on the bitterly cold afternoon of November 26, 1842, "
    "when a 28-year-old French priest, Rev. Edward Sorin, C.S.C., and seven companions, all members "
    "of the recently established Congregation of Holy Cross, took possession of 524 snow-covered acres "
    "that the Bishop of Vincennes had given them in the Indiana mission fields."
)

questions = [
    "When was the University of Notre Dame founded?",
    "Who founded the University of Notre Dame?",
    "Where is the University of Notre Dame located?",
]

for q in questions:
    result = qa.answer(question=q, context=passage)
    print(f"Q: {q}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")
    print("-" * 40)
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
