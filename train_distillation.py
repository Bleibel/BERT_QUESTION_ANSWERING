#!/usr/bin/env python3
"""Knowledge Distillation pipeline for training Micro-BERT.

This script trains a compact ~10M-parameter Student Micro-BERT model
using a fine-tuned BERT model as a Teacher. By matching soft probability
distributions (logits) from the Teacher via KL Divergence, the Student gains
significant accuracy boosts while remaining extremely lightweight.

Usage:
    python train_distillation.py --dataset squad --epochs 5 --batch_size 16 --fp16 --device 0
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from src.micro_bert import get_micro_bert_config, count_parameters
from train import prepare_train_features, load_json_dataset, load_squad_dataset


class DistillationTrainer(Trainer):
    """Custom Hugging Face Trainer for Knowledge Distillation."""

    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override loss computation to incorporate soft distillation loss."""
        # 1. Forward pass on Student
        outputs_student = model(**inputs)
        student_start_logits = outputs_student.start_logits
        student_end_logits = outputs_student.end_logits

        # Standard hard target cross-entropy loss
        loss_ce = outputs_student.loss

        # 2. Forward pass on Teacher (no gradient tracing)
        if self.teacher_model is not None:
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs.get("token_type_ids", None)
                )
                teacher_start_logits = outputs_teacher.start_logits
                teacher_end_logits = outputs_teacher.end_logits

            # 3. Soft target loss (KL Divergence with Temperature smoothing)
            p_start_student = F.log_softmax(student_start_logits / self.temperature, dim=-1)
            p_end_student = F.log_softmax(student_end_logits / self.temperature, dim=-1)

            p_start_teacher = F.softmax(teacher_start_logits / self.temperature, dim=-1)
            p_end_teacher = F.softmax(teacher_end_logits / self.temperature, dim=-1)

            loss_kl_start = F.kl_div(p_start_student, p_start_teacher, reduction="batchmean")
            loss_kl_end = F.kl_div(p_end_student, p_end_teacher, reduction="batchmean")
            loss_kl = loss_kl_start + loss_kl_end

            # 4. Joint loss formulation
            # Scaled by temperature squared to balance gradients
            loss = (1.0 - self.alpha) * loss_ce + self.alpha * (self.temperature ** 2) * loss_kl
        else:
            loss = loss_ce

        return (loss, outputs_student) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Distill Teacher BERT into Student Micro-BERT QA")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sample_squad.json",
        help="Path to SQuAD JSON or 'squad' for full HF SQuAD 1.1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/micro-bert-qa-distilled",
        help="Directory to save the distilled student model",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="csarron/bert-base-uncased-squad-v1",
        help="Teacher model path or Hugging Face repository id",
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
        default=5e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Distillation soft loss weighting coefficient (alpha)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Temperature smoothing factor (T) for soft targets",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Max sequence token length",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="Token stride for sliding-window chunking",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed-precision (FP16) training on CUDA",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device for training (-1 for CPU, 0+ for GPU)",
    )
    args = parser.parse_args()

    # Load dataset
    if args.dataset.lower() == "squad":
        from datasets import Dataset
        dataset = load_squad_dataset("train")
    else:
        if not os.path.exists(args.dataset):
            print(f"Dataset path not found: {args.dataset}")
            sys.exit(1)
        print(f"Loading dataset from {args.dataset}...")
        examples = load_json_dataset(args.dataset)
        print(f"Loaded {len(examples)} examples.")
        from datasets import Dataset
        dataset = Dataset.from_list(examples)

    # Initialize Tokenizer and models
    print(f"Loading Student Tokenizer and Model Configuration...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    student_config = get_micro_bert_config()
    student_model = BertForQuestionAnswering(student_config)

    print(f"Loading Teacher Model from {args.teacher_model}...")
    try:
        teacher_model = BertForQuestionAnswering.from_pretrained(args.teacher_model)
    except Exception as e:
        print(f"Failed to load Teacher Model: {str(e)}")
        print("Proceeding to train Student from scratch WITHOUT teacher distillation (standard fine-tuning).")
        teacher_model = None

    print("=" * 60)
    print("Knowledge Distillation Parameters")
    print("=" * 60)
    print(f"Teacher Model     : {args.teacher_model if teacher_model else 'None (CE Only)'}")
    print(f"Student Params    : {count_parameters(student_model):,}")
    print(f"Distillation Alpha: {args.alpha}")
    print(f"Temperature (T)   : {args.temperature}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Learning Rate     : {args.learning_rate}")
    print("=" * 60)

    # Tokenize features
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    for item in os.listdir(args.output_dir):
        item_path = os.path.join(args.output_dir, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        except Exception:
            pass

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * len(tokenized_dataset) / args.batch_size)),
        logging_steps=5,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
        fp16=args.fp16,
        no_cuda=(args.device < 0),
    )

    # Run distillation trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )

    print("Starting Distillation Training...")
    trainer.train()

    # Save final model checkpoints
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nDistilled Student model saved to {args.output_dir}")

    # Metadata file
    meta = {
        "model_type": "micro-bert-qa-distilled",
        "student_parameters": count_parameters(student_model),
        "distillation": {
            "teacher_model": args.teacher_model if teacher_model else None,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
    }
    with open(os.path.join(args.output_dir, "distill_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Distillation pipeline completed!")


if __name__ == "__main__":
    main()
