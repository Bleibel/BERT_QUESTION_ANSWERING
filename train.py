#!/usr/bin/env python3
"""Fine-tune a custom Micro-BERT for extractive Question Answering.

This script trains our ~1.8M-parameter Micro-BERT on SQuAD-format data
using the Hugging Face Trainer API.

Quick start (sample data, CPU, ~30 seconds):
    python train.py

Train on full SQuAD 1.1 (CPU, ~20-40 minutes):
    python train.py --dataset squad --epochs 2 --batch_size 8

Train on custom JSON:
    python train.py --dataset data/sample_squad.json --epochs 5
"""

import argparse
import json
import os
import sys

from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from src.micro_bert import get_micro_bert_config, count_parameters


def load_json_dataset(path: str):
    """Load a SQuAD-format JSON file into a list of examples."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                examples.append({
                    "id": qa["id"],
                    "title": article.get("title", ""),
                    "context": context,
                    "question": qa["question"],
                    "answers": {
                        "text": [a["text"] for a in answers],
                        "answer_start": [a["answer_start"] for a in answers],
                    },
                })
    return examples


def load_squad_dataset(split="train"):
    """Load SQuAD 1.1 from the Hugging Face datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("The 'datasets' library is required for full SQuAD training.")
        print("Install it with: pip install datasets")
        sys.exit(1)

    print(f"Downloading SQuAD 1.1 ({split})...")
    dataset = load_dataset("squad", split=split)
    return dataset


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """Tokenize examples with sliding-window chunking for training."""
    # When using HF datasets, examples is a dict of lists
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end = answer_start + len(answer_text)

        # Find token positions that map to the answer span
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # If answer is not within the span, set to [CLS]
        if not (
            offsets[token_start_index][0] <= answer_start
            and offsets[token_end_index][1] >= answer_end
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Move token_start_index to answer_start
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= answer_start
            ):
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            # Move token_end_index to answer_end
            while offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Micro-BERT for Question Answering"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sample_squad.json",
        help="Path to SQuAD JSON or 'squad' for full HF SQuAD 1.1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/micro-bert-qa",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Max token length per example",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="Stride for sliding window during tokenization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    if args.dataset.lower() == "squad":
        from datasets import Dataset
        raw_data = load_squad_dataset("train")
        # Convert to standard format for our prep function
        dataset = raw_data
    else:
        if not os.path.exists(args.dataset):
            print(f"Dataset not found: {args.dataset}")
            sys.exit(1)
        print(f"Loading dataset from {args.dataset}...")
        examples = load_json_dataset(args.dataset)
        print(f"Loaded {len(examples)} examples.")
        from datasets import Dataset
        dataset = Dataset.from_list(examples)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = get_micro_bert_config()
    model = BertForQuestionAnswering(config)

    print("=" * 60)
    print("Micro-BERT Configuration")
    print("=" * 60)
    print(f"Hidden size      : {config.hidden_size}")
    print(f"Layers           : {config.num_hidden_layers}")
    print(f"Attention heads  : {config.num_attention_heads}")
    print(f"Intermediate     : {config.intermediate_size}")
    print(f"Max positions    : {config.max_position_embeddings}")
    print(f"Total parameters : {count_parameters(model):,}")
    print("=" * 60)

    # Tokenize
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Clean output dir if it exists to avoid resume conflicts
    if os.path.exists(args.output_dir):
        import shutil
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * len(tokenized_dataset) / args.batch_size)),
        # logging_dir deprecated in v5.2; logs go to output_dir/logs by default
        logging_steps=5,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # Save a small metadata file
    meta = {
        "model_type": "micro-bert-qa",
        "total_parameters": count_parameters(model),
        "config": config.to_dict(),
        "training": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
