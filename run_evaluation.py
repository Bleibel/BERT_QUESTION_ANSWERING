#!/usr/bin/env python3
"""Run evaluation of BERT QA on a sample dataset.

Usage:
    python run_evaluation.py [--dataset path/to/squad.json] [--output results/eval_results.json]

If no dataset is provided, uses the built-in sample dataset.
"""

import argparse
import json
import os
import time

from src.model import BERTQA
from src.evaluate import evaluate_predictions, print_metrics
from src.dataset import load_squad_data, get_sample_dataset, save_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BERT Question Answering model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to SQuAD-format JSON dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="results/predictions.json",
        help="Path to save raw predictions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (default: micro-bert checkpoint if available, else BERT-large)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=384,
        help="Max tokens per chunk",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Sliding window stride",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of examples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device for model inference (-1 for CPU, 0+ for GPU)",
    )
    args = parser.parse_args()

    # Load dataset
    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        examples = load_squad_data(args.dataset)
    else:
        print("Using built-in sample dataset...")
        examples = get_sample_dataset()

    if args.max_examples:
        examples = examples[: args.max_examples]

    print(f"Total examples to evaluate: {len(examples)}")

    # Initialize model
    print("Loading model...")
    model = BERTQA(model_name=args.model, device=args.device)
    info = model.info()
    print(f"  Model : {info['model_name']}")
    print(f"  Params: {info['num_parameters']:,}")

    # Run predictions
    predictions = {}
    references = {}
    print("Running inference...")
    start_time = time.time()

    for i, ex in enumerate(examples):
        result = model.answer(
            question=ex["question"],
            context=ex["context"],
            chunk_size=args.chunk_size,
            stride=args.stride,
        )
        predictions[ex["id"]] = result["answer"]
        references[ex["id"]] = ex["answer_text"]

        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            print(
                f"  Processed {i + 1}/{len(examples)} examples "
                f"({avg_time:.2f}s avg)"
            )

    total_time = time.time() - start_time
    print(f"\nInference complete in {total_time:.2f}s")

    # Evaluate
    metrics = evaluate_predictions(predictions, references)
    print_metrics(metrics)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.predictions) or ".", exist_ok=True)

    save_predictions(predictions, args.predictions)
    print(f"\nPredictions saved to {args.predictions}")

    # Save detailed results
    results = {
        "config": {
            "model": info["model_name"],
            "num_parameters": info["num_parameters"],
            "is_micro": info["is_micro"],
            "chunk_size": args.chunk_size,
            "stride": args.stride,
            "num_examples": len(examples),
            "total_inference_time": total_time,
        },
        "metrics": {
            "exact_match": metrics["exact_match"],
            "f1": metrics["f1"],
        },
        "per_example": metrics["per_example"],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
