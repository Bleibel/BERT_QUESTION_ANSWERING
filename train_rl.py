#!/usr/bin/env python3
"""Reinforcement Learning (Policy Gradient) training for Micro-BERT QA.

This script implements REINFORCE to fine-tune the QA model using
reward signals (F1, EM) rather than ground-truth labels.

Usage:
    python train_rl.py --checkpoint checkpoints/micro-bert-qa --epochs 10

Requirements:
    - A pre-trained or supervised checkpoint to start from
    - The model predicts spans; rewards are computed against ground truth
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForQuestionAnswering

from src.dataset import get_sample_dataset, load_squad_data
from src.evaluate import exact_match_score, f1_score
from src.micro_bert import count_parameters


def compute_rewards(predictions: List[str], references: List[str]) -> torch.Tensor:
    """Compute rewards for a batch of predictions.

    Default: combined reward = 0.5 * EM + 0.5 * F1
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        em = exact_match_score(pred, ref)
        f1 = f1_score(pred, ref)
        # Combined reward with small length penalty to avoid spurious spans
        reward = 0.5 * em + 0.5 * f1
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)


def sample_span(start_logits: torch.Tensor, end_logits: torch.Tensor, mask: torch.Tensor, max_answer_len: int = 100):
    """Sample answer spans using the policy distribution.

    Returns sampled (start_idx, end_idx) and their log probabilities.
    """
    batch_size, seq_len = start_logits.shape

    # Apply mask: set invalid positions to -inf
    start_logits = start_logits.masked_fill(~mask.bool(), float('-inf'))
    end_logits = end_logits.masked_fill(~mask.bool(), float('-inf'))

    # Get start probabilities
    start_probs = F.softmax(start_logits, dim=-1)  # (batch, seq_len)

    sampled_spans = []
    log_probs = []

    for b in range(batch_size):
        # Sample start position
        start_dist = torch.distributions.Categorical(start_probs[b])
        start_idx = start_dist.sample()
        start_log_prob = start_dist.log_prob(start_idx)

        # Create end mask: end must be >= start and <= start + max_answer_len
        end_mask = mask[b].clone()
        valid_end = torch.zeros(seq_len, dtype=torch.bool, device=end_logits.device)
        end_limit = min(start_idx.item() + max_answer_len, seq_len)
        valid_end[start_idx:end_limit] = True
        end_mask = end_mask & valid_end

        # Mask invalid end positions
        masked_end_logits = end_logits[b].masked_fill(~end_mask.bool(), float('-inf'))
        end_probs = F.softmax(masked_end_logits, dim=-1)

        # Sample end position
        end_dist = torch.distributions.Categorical(end_probs)
        end_idx = end_dist.sample()
        end_log_prob = end_dist.log_prob(end_idx)

        # Combined log probability of this (start, end) pair
        log_prob = start_log_prob + end_log_prob

        sampled_spans.append((start_idx.item(), end_idx.item()))
        log_probs.append(log_prob)

    return sampled_spans, torch.stack(log_probs)


def decode_spans(input_ids: torch.Tensor, spans: List[Tuple[int, int]], tokenizer) -> List[str]:
    """Decode token spans back to text strings."""
    answers = []
    for b, (start, end) in enumerate(spans):
        tokens = input_ids[b][start:end + 1]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        answers.append(text)
    return answers


def prepare_batch(examples: List[dict], tokenizer, max_length: int = 384, device: str = "cpu"):
    """Tokenize a batch of examples for RL training."""
    questions = [ex["question"] for ex in examples]
    contexts = [ex["context"] for ex in examples]
    references = [ex["answer_text"] for ex in examples]

    encoding = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Create mask: 1 for valid context tokens, 0 for question/padding
    # Find first [SEP] (end of question)
    sep_token_id = tokenizer.sep_token_id
    masks = []
    for i in range(encoding["input_ids"].shape[0]):
        seq = encoding["input_ids"][i]
        sep_positions = (seq == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) >= 1:
            question_end = sep_positions[0].item() + 1
        else:
            question_end = 0

        mask = encoding["attention_mask"][i].clone()
        mask[:question_end] = 0  # Mask out question + [CLS]
        masks.append(mask)

    mask_tensor = torch.stack(masks).to(device)

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).to(device),
        "mask": mask_tensor,
        "references": references,
    }


def train_rl(
    model,
    tokenizer,
    dataset,
    optimizer,
    epochs: int = 10,
    batch_size: int = 4,
    max_length: int = 384,
    baseline_decay: float = 0.9,
    device: str = "cpu",
    output_dir: str = "checkpoints/micro-bert-qa-rl",
):
    """Train the QA model using REINFORCE with baseline."""
    model.to(device)
    model.train()

    os.makedirs(output_dir, exist_ok=True)

    # Running baseline for variance reduction
    baseline = 0.0

    print("=" * 60)
    print("RL Training (REINFORCE)")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Baseline decay: {baseline_decay}")
    print(f"Device: {device}")
    print("=" * 60)

    num_batches = math.ceil(len(dataset) / batch_size)

    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        epoch_losses = []

        for i in range(0, len(dataset), batch_size):
            batch_examples = dataset[i:i + batch_size]
            batch = prepare_batch(batch_examples, tokenizer, max_length, device)

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Sample spans from policy
            sampled_spans, log_probs = sample_span(
                start_logits, end_logits, batch["mask"], max_answer_len=100
            )

            # Decode to text
            predictions = decode_spans(batch["input_ids"], sampled_spans, tokenizer)

            # Compute rewards
            rewards = compute_rewards(predictions, batch["references"])
            rewards = rewards.to(device)

            # Update baseline (exponential moving average)
            mean_reward = rewards.mean().item()
            baseline = baseline_decay * baseline + (1 - baseline_decay) * mean_reward

            # REINFORCE loss: -log_prob * (reward - baseline)
            advantages = rewards - baseline
            loss = -(log_probs * advantages).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_rewards.extend(rewards.tolist())
            epoch_losses.append(loss.item())

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        print(f"Epoch {epoch}/{epochs} | Avg Reward: {avg_reward:.4f} | Avg Loss: {avg_loss:.4f} | Baseline: {baseline:.4f}")

    # Save final RL checkpoint
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    meta = {
        "training_type": "reinforcement_learning",
        "algorithm": "REINFORCE",
        "epochs": epochs,
        "batch_size": batch_size,
        "final_baseline": baseline,
        "final_avg_reward": avg_reward,
    }
    with open(os.path.join(output_dir, "rl_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nRL model saved to {output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="RL Training for Micro-BERT QA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/micro-bert-qa", help="Starting checkpoint")
    parser.add_argument("--dataset", type=str, default="data/sample_squad.json", help="Dataset path or 'squad'")
    parser.add_argument("--output_dir", type=str, default="checkpoints/micro-bert-qa-rl", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="RL training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = BertForQuestionAnswering.from_pretrained(args.checkpoint)
    print(f"Parameters: {count_parameters(model):,}")

    # Load dataset
    if args.dataset.lower() == "squad":
        from datasets import load_dataset
        raw = load_dataset("rajpurkar/squad", split="train")
        # Convert to our format
        dataset = []
        for ex in raw:
            dataset.append({
                "id": ex["id"],
                "context": ex["context"],
                "question": ex["question"],
                "answer_text": ex["answers"]["text"][0],
            })
    else:
        dataset = load_squad_data(args.dataset)

    print(f"Dataset size: {len(dataset)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    train_rl(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
