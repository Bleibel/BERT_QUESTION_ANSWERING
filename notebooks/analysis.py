"""Analysis script for BERT QA evaluation results.

Run after evaluation to generate summary statistics and error analysis.
"""

import json
import os


def load_results(path="results/eval_results.json"):
    """Load evaluation results from JSON."""
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        print("Run: python run_evaluation.py")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(results):
    """Print detailed analysis of evaluation results."""
    if results is None:
        return

    config = results.get("config", {})
    metrics = results.get("metrics", {})
    examples = results.get("per_example", [])

    print("=" * 60)
    print("BERT QA Evaluation Analysis")
    print("=" * 60)
    print(f"Model           : {config.get('model', 'N/A')}")
    print(f"Chunk Size      : {config.get('chunk_size', 'N/A')}")
    print(f"Stride          : {config.get('stride', 'N/A')}")
    print(f"Total Examples  : {config.get('num_examples', len(examples))}")
    print(f"Inference Time  : {config.get('total_inference_time', 0):.2f}s")
    print("-" * 60)
    print(f"Exact Match     : {metrics.get('exact_match', 0):.2f}%")
    print(f"F1 Score        : {metrics.get('f1', 0):.2f}%")
    print("=" * 60)

    # Error analysis
    correct = [ex for ex in examples if ex.get("exact_match", 0) == 1.0]
    incorrect = [ex for ex in examples if ex.get("exact_match", 0) == 0.0]

    print(f"\nCorrect predictions  : {len(correct)}")
    print(f"Incorrect predictions: {len(incorrect)}")

    if incorrect:
        print("\n--- Error Analysis (Incorrect Examples) ---")
        for ex in incorrect:
            print(f"\nID      : {ex['id']}")
            print(f"Pred    : {ex['prediction']}")
            print(f"Truth   : {ex['ground_truth']}")
            print(f"F1      : {ex['f1']:.2f}")
            print("-" * 40)

    # Confidence distribution
    if examples:
        f1s = [ex["f1"] for ex in examples]
        avg_f1 = sum(f1s) / len(f1s)
        print(f"\nAverage F1 (raw)    : {avg_f1:.4f}")


if __name__ == "__main__":
    results = load_results()
    analyze(results)
