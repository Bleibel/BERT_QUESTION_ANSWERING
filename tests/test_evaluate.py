"""Unit tests for evaluation metrics and utilities."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import exact_match_score, f1_score, evaluate_predictions
from src.utils import normalize_text, sliding_window_chunks


def test_normalize_text():
    assert normalize_text("The quick brown fox!") == "quick brown fox"
    assert normalize_text("  A  test  ") == "test"


def test_exact_match():
    assert exact_match_score("Berlin", "Berlin") == 1.0
    assert exact_match_score("berlin", "Berlin") == 1.0
    assert exact_match_score("Berlin ", "Berlin") == 1.0
    assert exact_match_score("Paris", "Berlin") == 0.0


def test_f1_score():
    assert f1_score("the quick brown fox", "the quick brown fox") == 1.0
    assert f1_score("quick fox", "the quick brown fox") > 0.0
    assert f1_score("", "something") == 0.0
    assert f1_score("", "") == 1.0


def test_evaluate_predictions():
    preds = {"a": "Berlin", "b": "Paris"}
    refs = {"a": "Berlin", "b": "Berlin"}
    metrics = evaluate_predictions(preds, refs)
    assert metrics["exact_match"] == 50.0
    assert metrics["total"] == 2


def test_sliding_window_chunks():
    text = "This is a test passage for chunking. " * 20
    chunks = sliding_window_chunks(text, max_length=100, stride=50, tokenizer=None)
    assert len(chunks) > 0
    assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)


if __name__ == "__main__":
    test_normalize_text()
    test_exact_match()
    test_f1_score()
    test_evaluate_predictions()
    test_sliding_window_chunks()
    print("All tests passed!")
