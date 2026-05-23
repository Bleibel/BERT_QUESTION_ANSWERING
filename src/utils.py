"""Utility functions for text preprocessing and chunking."""

import re
import string
from typing import List, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison.

    Lowercases, removes punctuation, articles, and extra whitespace.
    Based on SQuAD evaluation script.
    """
    text = text.lower()
    text = remove_articles(text)
    text = fix_whitespace(text)
    text = remove_punc(text)
    return text


def remove_articles(text: str) -> str:
    """Remove English articles."""
    return re.sub(r"\b(a|an|the)\b", " ", text)


def remove_punc(text: str) -> str:
    """Remove punctuation."""
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def fix_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return " ".join(text.split())


def sliding_window_chunks(
    text: str,
    max_length: int = 384,
    stride: int = 128,
    tokenizer=None,
) -> List[Tuple[str, int]]:
    """Split text into overlapping chunks using a sliding window.

    Args:
        text: The input passage.
        max_length: Maximum token length per chunk (including question).
        stride: Number of tokens to slide the window by.
        tokenizer: Hugging Face tokenizer for accurate token-based chunking.

    Returns:
        List of (chunk_text, start_char_offset) tuples.
    """
    if tokenizer is None:
        # Fallback to character-based chunking
        return _char_sliding_window(text, max_length, stride)

    # Token-based sliding window for better alignment with model
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tokens=True,
    )
    input_ids = tokens["input_ids"]
    offset_mapping = tokens["offset_mapping"]

    chunks = []
    start_idx = 0
    while start_idx < len(input_ids):
        end_idx = min(start_idx + max_length, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]

        # Decode chunk back to text
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Determine character offset of this chunk in original text
        if start_idx < len(offset_mapping):
            char_start = offset_mapping[start_idx][0]
        else:
            char_start = 0

        chunks.append((chunk_text, char_start))

        if end_idx >= len(input_ids):
            break
        start_idx += stride

    return chunks


def _char_sliding_window(text: str, max_chars: int, stride_chars: int) -> List[Tuple[str, int]]:
    """Character-based sliding window fallback."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at a sentence boundary if possible
        if end < len(text):
            for delim in ".\n":
                pos = text.rfind(delim, start, end)
                if pos != -1 and pos > start + max_chars // 2:
                    end = pos + 1
                    break
        chunk = text[start:end]
        chunks.append((chunk, start))
        if end >= len(text):
            break
        start += stride_chars
    return chunks


def find_best_answer(
    answers: List[dict],
) -> dict:
    """Select the best answer from multiple chunk predictions.

    Uses confidence score and answer length heuristics.

    Args:
        answers: List of answer dicts with keys: answer, score, start, end.

    Returns:
        The best answer dict.
    """
    if not answers:
        return {"answer": "", "score": 0.0, "start": 0, "end": 0}

    # Primary: highest confidence score
    # Secondary: prefer non-empty answers
    # Tertiary: prefer shorter answers (less likely to be spurious)
    def sort_key(ans):
        score = ans.get("score", 0.0)
        is_empty = 1 if not ans.get("answer", "").strip() else 0
        length = len(ans.get("answer", ""))
        return (score, -is_empty, -1.0 / (1.0 + length))

    return max(answers, key=sort_key)
