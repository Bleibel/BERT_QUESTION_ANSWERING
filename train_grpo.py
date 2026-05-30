#!/usr/bin/env python3
"""Group Relative Policy Optimization (GRPO) for Micro-BERT QA.

This implements DeepSeek-R1's GRPO algorithm for extractive question answering.
Unlike PPO, GRPO removes the critic network entirely — the baseline is simply
the average reward of a sampled group of answers.

References:
    - Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical
      Reasoning in Open Language Models. (Introduced GRPO)
    - DeepSeek-R1 (2025). https://github.com/deepseek-ai/DeepSeek-R1

Usage:
    # 1. Train supervised checkpoint first
    python train.py --dataset squad --epochs 10 --batch_size 32 --fp16 --device 0

    # 2. GRPO fine-tuning
    python train_grpo.py \
        --checkpoint checkpoints/micro-bert-qa \
        --dataset squad \
        --epochs 5 \
        --group_size 8 \
        --device 0
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForQuestionAnswering

from src.dataset import load_squad_data_hf
from src.evaluate import exact_match_score, f1_score
from src.micro_bert import count_parameters


def compute_rewards(predictions: List[str], references: List[str], length_penalty_coeff: float = 0.002) -> torch.Tensor:
    """Compute rewards for a batch of predictions.

    Reward = 0.5 * EM + 0.5 * F1 - length_penalty
    """
    rewards = []
    for pred, ref in zip(predictions, references):
        em = exact_match_score(pred, ref)
        f1 = f1_score(pred, ref)
        word_count = len(pred.split())
        penalty = length_penalty_coeff * word_count
        reward = 0.5 * em + 0.5 * f1 - penalty
        reward = max(-0.2, reward)
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)


def sample_spans_group(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    mask: torch.Tensor,
    group_size: int = 8,
    max_answer_len: int = 100,
) -> Tuple[List[List[Tuple[int, int]]], List[torch.Tensor]]:
    """Sample G answer spans per example from the policy distribution.

    Returns:
        spans: List of length batch_size, each containing G (start, end) tuples
        log_probs: List of length batch_size, each a tensor of G log_probs
    """
    batch_size, seq_len = start_logits.shape
    device = start_logits.device

    # Mask invalid positions
    start_logits = start_logits.masked_fill(~mask.bool(), float('-inf'))
    end_logits = end_logits.masked_fill(~mask.bool(), float('-inf'))

    start_probs = F.softmax(start_logits, dim=-1)

    all_spans = []
    all_log_probs = []

    for b in range(batch_size):
        spans = []
        log_probs = []

        for _ in range(group_size):
            # Sample start
            start_dist = torch.distributions.Categorical(start_probs[b])
            start_idx = start_dist.sample()
            start_log_prob = start_dist.log_prob(start_idx)

            # Create end mask
            end_limit = min(start_idx.item() + max_answer_len, seq_len)
            valid_end = torch.zeros(seq_len, dtype=torch.bool, device=device)
            valid_end[start_idx:end_limit] = True
            end_mask = mask[b] & valid_end

            masked_end_logits = end_logits[b].masked_fill(~end_mask.bool(), float('-inf'))
            end_probs = F.softmax(masked_end_logits, dim=-1)

            # Sample end
            end_dist = torch.distributions.Categorical(end_probs)
            end_idx = end_dist.sample()
            end_log_prob = end_dist.log_prob(end_idx)

            log_prob = start_log_prob + end_log_prob
            spans.append((start_idx.item(), end_idx.item()))
            log_probs.append(log_prob)

        all_spans.append(spans)
        all_log_probs.append(torch.stack(log_probs))

    return all_spans, all_log_probs


def decode_spans(input_ids: torch.Tensor, spans: List[Tuple[int, int]], tokenizer) -> List[str]:
    """Decode token spans back to text strings."""
    answers = []
    for start, end in spans:
        tokens = input_ids[start:end + 1]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        answers.append(text)
    return answers


def prepare_batch(examples: List[dict], tokenizer, max_length: int = 384, device: str = "cpu"):
    """Tokenize a batch of examples for GRPO training."""
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

    sep_token_id = tokenizer.sep_token_id
    masks = []
    for i in range(encoding["input_ids"].shape[0]):
        seq = encoding["input_ids"][i]
        sep_positions = (seq == sep_token_id).nonzero(as_tuple=True)[0]
        question_end = sep_positions[0].item() + 1 if len(sep_positions) >= 1 else 0
        mask = encoding["attention_mask"][i].clone()
        mask[:question_end] = 0
        masks.append(mask)

    mask_tensor = torch.stack(masks).to(device)

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).to(device),
        "mask": mask_tensor,
        "references": references,
    }


def train_grpo(
    model,
    ref_model,
    tokenizer,
    dataset,
    optimizer,
    epochs: int = 5,
    batch_size: int = 4,
    group_size: int = 8,
    max_length: int = 384,
    epsilon: float = 0.2,
    beta: float = 0.01,
    device: str = "cpu",
    output_dir: str = "checkpoints/micro-bert-qa-grpo",
):
    """Train the QA model using GRPO (Group Relative Policy Optimization).

    Args:
        model: The policy model (trainable).
        ref_model: The reference model (frozen, from supervised checkpoint).
        tokenizer: Tokenizer.
        dataset: List of examples.
        optimizer: Optimizer.
        epochs: Number of training epochs.
        batch_size: Number of examples per batch.
        group_size: Number of sampled spans per example (G).
        max_length: Max sequence length.
        epsilon: PPO clipping parameter.
        beta: KL penalty coefficient.
        device: Device string.
        output_dir: Where to save the checkpoint.
    """
    model.to(device)
    model.train()
    ref_model.to(device)
    ref_model.eval()

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Training (Group Relative Policy Optimization)")
    print("=" * 60)
    print(f"Epochs:     {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Group size: {group_size}")
    print(f"Epsilon:    {epsilon}")
    print(f"KL beta:    {beta}")
    print(f"Device:     {device}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        epoch_losses = []

        # Shuffle dataset each epoch
        indices = torch.randperm(len(dataset)).tolist()

        for i in range(0, len(dataset), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_examples = [dataset[idx] for idx in batch_idx]
            batch = prepare_batch(batch_examples, tokenizer, max_length, device)

            # === Policy forward (trainable) ===
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # === Reference forward (frozen) ===
            with torch.no_grad():
                ref_outputs = ref_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                )
                ref_start_logits = ref_outputs.start_logits
                ref_end_logits = ref_outputs.end_logits

            # === Sample G spans per example ===
            sampled_spans, log_probs = sample_spans_group(
                start_logits, end_logits, batch["mask"], group_size=group_size
            )

            # Also get reference log_probs for KL penalty
            with torch.no_grad():
                ref_start_probs = F.softmax(ref_start_logits.masked_fill(~batch["mask"].bool(), float('-inf')), dim=-1)
                ref_end_probs = F.softmax(ref_end_logits.masked_fill(~batch["mask"].bool(), float('-inf')), dim=-1)

            # === Compute rewards per group ===
            batch_losses = []
            for b in range(len(batch_examples)):
                # Decode all G spans for this example
                predictions = decode_spans(batch["input_ids"][b].unsqueeze(0), sampled_spans[b], tokenizer)
                references = [batch["references"][b]] * group_size

                rewards = compute_rewards(predictions, references)
                rewards = rewards.to(device)

                # Group relative baseline: mean of this group's rewards
                group_mean = rewards.mean()
                advantages = rewards - group_mean

                epoch_rewards.extend(rewards.tolist())

                # Get reference log_probs for KL
                ref_log_probs_b = []
                for g, (s, e) in enumerate(sampled_spans[b]):
                    ref_s_logp = torch.log(ref_start_probs[b][s] + 1e-10)
                    ref_e_logp = torch.log(ref_end_probs[b][e] + 1e-10)
                    ref_log_probs_b.append(ref_s_logp + ref_e_logp)
                ref_log_probs_b = torch.stack(ref_log_probs_b)

                # PPO-style clipping
                ratio = torch.exp(log_probs[b] - ref_log_probs_b.detach())
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

                policy_loss = -torch.min(
                    ratio * advantages,
                    clipped_ratio * advantages
                ).mean()

                # KL penalty: prevent deviation from reference policy
                kl_penalty = (log_probs[b] - ref_log_probs_b).mean()

                loss_b = policy_loss + beta * kl_penalty
                batch_losses.append(loss_b)

            loss = torch.stack(batch_losses).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"Epoch {epoch}/{epochs} | Avg Reward: {avg_reward:.4f} | Avg Loss: {avg_loss:.4f}")

    # Save final GRPO checkpoint
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "training_type": "reinforcement_learning",
        "algorithm": "GRPO",
        "epochs": epochs,
        "batch_size": batch_size,
        "group_size": group_size,
        "epsilon": epsilon,
        "beta": beta,
        "final_avg_reward": avg_reward,
    }
    with open(os.path.join(output_dir, "grpo_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGRPO model saved to {output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Micro-BERT QA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/micro-bert-qa", help="Supervised checkpoint to start from")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset path or 'squad' for full HF SQuAD")
    parser.add_argument("--output_dir", type=str, default="checkpoints/micro-bert-qa-grpo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="GRPO training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of questions)")
    parser.add_argument("--group_size", type=int, default=8, help="Number of sampled spans per question (G)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy model (trainable)
    print(f"Loading policy checkpoint from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = BertForQuestionAnswering.from_pretrained(args.checkpoint)
    print(f"Policy parameters: {count_parameters(model):,}")

    # Load reference model (frozen copy)
    print("Loading reference model (frozen)...")
    ref_model = BertForQuestionAnswering.from_pretrained(args.checkpoint)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # Load dataset
    if args.dataset.lower() == "squad":
        dataset = load_squad_data_hf(split="train")
    else:
        dataset = load_squad_data(args.dataset)

    print(f"Dataset size: {len(dataset)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    train_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizer=optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_length=args.max_length,
        epsilon=args.epsilon,
        beta=args.beta,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
