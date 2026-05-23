"""Evaluation metrics for Question Answering."""

import collections
import json
import re
import string
from typing import Dict, List

from src.utils import normalize_text


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match (EM) score.

    Returns 1.0 if normalized prediction matches normalized ground truth,
    0.0 otherwise.
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_predictions(
    predictions: Dict[str, str],
    references: Dict[str, str],
) -> dict:
    """Evaluate a set of predictions against ground truth references.

    Args:
        predictions: Dict mapping example_id -> predicted answer string.
        references: Dict mapping example_id -> ground truth answer string.

    Returns:
        Dict with exact_match, f1, and per-example scores.
    """
    total_em = 0.0
    total_f1 = 0.0
    per_example = []

    for ex_id, pred in predictions.items():
        gt = references.get(ex_id, "")
        em = exact_match_score(pred, gt)
        f1 = f1_score(pred, gt)
        total_em += em
        total_f1 += f1
        per_example.append({
            "id": ex_id,
            "prediction": pred,
            "ground_truth": gt,
            "exact_match": em,
            "f1": f1,
        })

    count = len(predictions)
    return {
        "exact_match": 100.0 * total_em / count if count > 0 else 0.0,
        "f1": 100.0 * total_f1 / count if count > 0 else 0.0,
        "total": count,
        "per_example": per_example,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total Examples : {metrics['total']}")
    print(f"Exact Match    : {metrics['exact_match']:.2f}%")
    print(f"F1 Score       : {metrics['f1']:.2f}%")
    print("=" * 50)
