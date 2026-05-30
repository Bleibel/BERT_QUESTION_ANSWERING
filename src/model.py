"""BERT Question Answering model wrapper.

Supports both standard Hugging Face models and our custom Micro-BERT
fine-tuned checkpoint. Uses manual inference instead of the pipeline
API for maximum compatibility and control.
"""

import os
import math
from typing import List, Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.utils import find_best_answer
from src.micro_bert import get_micro_bert_config, count_parameters


class ValueHead(torch.nn.Module):
    """Critic value head for PPO RL training.

    Estimates the expected reward (value) of a given (question, passage) state
    by processing the [CLS] token representation.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.dense = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate for each example in batch.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            values: [batch_size]
        """
        cls_token = hidden_states[:, 0]  # [CLS]
        cls_token = self.dropout(cls_token)
        value = self.dense(cls_token).squeeze(-1)
        return value


DEFAULT_MICRO_CHECKPOINT = "checkpoints/micro-bert-qa"


class BERTQA:
    """BERT-based Question Answering system with sliding-window chunking."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        max_answer_length: int = 100,
        null_threshold: float = 0.15,
    ):
        """Initialize the QA pipeline.

        Args:
            model_name: Hugging Face model id or local path. If None, uses
                        the Micro-BERT checkpoint.
            device: -1 for CPU, >=0 for GPU device id.
            max_answer_length: Maximum length of extracted answer span.
            null_threshold: Confidence threshold below which answers are marked unanswerable.
        """
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.max_answer_length = max_answer_length
        self.null_threshold = null_threshold

        # Resolve model path
        if model_name is None:
            model_name = DEFAULT_MICRO_CHECKPOINT
            self.is_micro = True
        else:
            self.is_micro = "micro-bert" in model_name.lower() or "micro" in model_name.lower()

        self.model_name = model_name

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # FlashAttention (SDPA) integration
        kwargs = {}
        if self.device.type == "cuda":
            try:
                # attn_implementation="sdpa" requires newer transformers package
                kwargs["attn_implementation"] = "sdpa"
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, **kwargs)
                print("[BERTQA] Enabled FlashAttention (SDPA) successfully.")
            except Exception:
                # Fallback
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
        self.model.to(self.device)
        self.model.eval()

        self.num_parameters = count_parameters(self.model)

    def _encode_sliding_window(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        stride: int = 128,
    ) -> List[dict]:
        """Tokenize question+context with sliding-window chunking.

        Returns a list of encoding dicts, each representing one chunk.
        """
        encoding = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        num_windows = encoding["input_ids"].shape[0]
        windows = []
        for i in range(num_windows):
            window = {k: v[i].unsqueeze(0).to(self.device) for k, v in encoding.items()}
            windows.append(window)
        return windows

    def _extract_answer_from_window(
        self,
        window: dict,
        context: str,
        question_length: int,
    ) -> Optional[dict]:
        """Run inference on a single window and extract the best answer span."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=window["input_ids"],
                attention_mask=window["attention_mask"],
                token_type_ids=window.get("token_type_ids"),
            )

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Mask out question tokens and padding
        sequence_ids = window["input_ids"][0] == self.tokenizer.cls_token_id
        # Actually, we can use attention mask
        mask = window["attention_mask"][0].bool()
        # Also mask out question tokens: find first sep token
        sep_indices = (window["input_ids"][0] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_indices) >= 1:
            question_end = sep_indices[0].item() + 1
        else:
            question_end = 0

        # Apply mask: valid positions are after question_end and before padding
        valid_mask = mask.clone()
        valid_mask[:question_end] = False

        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        # Find best start/end pair
        best_score = -float("inf")
        best_start = 0
        best_end = 0

        start_logits_masked = start_logits.clone()
        end_logits_masked = end_logits.clone()
        start_logits_masked[~valid_mask] = -1e9
        end_logits_masked[~valid_mask] = -1e9

        # Greedy approach: pick top-k starts and check corresponding ends
        num_valid = valid_mask.sum().item()
        if num_valid == 0:
            return None
        top_k = min(20, num_valid)
        top_start_indices = torch.topk(start_logits_masked, top_k).indices.tolist()

        for start_idx in top_start_indices:
            # End must be >= start and <= start + max_answer_length
            end_limit = min(start_idx + self.max_answer_length, len(end_logits_masked))
            end_candidates = end_logits_masked[start_idx:end_limit]
            if end_candidates.numel() == 0:
                continue
            best_local_end = torch.argmax(end_candidates).item() + start_idx
            score = start_logits_masked[start_idx].item() + end_logits_masked[best_local_end].item()
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = best_local_end

        if best_start == 0 and best_end == 0:
            return None

        # Decode answer
        answer_tokens = window["input_ids"][0][best_start:best_end + 1]
        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        if not answer_text:
            return None

        # Map to character offsets in original context
        offset_mapping = window["offset_mapping"][0].cpu().tolist()
        char_start = offset_mapping[best_start][0]
        char_end = offset_mapping[best_end][1]

        # Clamp to context length
        char_start = max(0, min(char_start, len(context)))
        char_end = max(0, min(char_end, len(context)))

        # Validate decoded text matches context slice
        context_slice = context[char_start:char_end]
        if answer_text not in context_slice and context_slice not in answer_text:
            # Try to find answer_text near char_start
            search_start = max(0, char_start - 20)
            search_end = min(len(context), char_end + 20)
            pos = context.find(answer_text, search_start, search_end)
            if pos != -1:
                char_start = pos
                char_end = pos + len(answer_text)

        # Confidence as average probability
        start_conf = start_probs[best_start].item()
        end_conf = end_probs[best_end].item()
        confidence = (start_conf + end_conf) / 2.0

        if confidence < self.null_threshold:
            return {
                "answer": "",
                "score": confidence,
                "start": 0,
                "end": 0,
            }

        return {
            "answer": answer_text,
            "score": confidence,
            "start": char_start,
            "end": char_end,
        }

    def answer(
        self,
        question: str,
        context: str,
        chunk_size: int = 384,
        stride: int = 128,
        top_k: int = 1,
    ) -> dict:
        """Answer a question given a context passage.

        For long contexts, a sliding window approach is used.

        Args:
            question: The question string.
            context: The passage string.
            chunk_size: Max tokens per chunk.
            stride: Sliding window stride in tokens.
            top_k: Number of top answers to return.

        Returns:
            Dict with keys: answer, score, start, end.
            If top_k > 1, returns a list under key "answers".
        """
        if not context or not question:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        # Check if context fits in one chunk
        test_encoding = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=chunk_size,
            return_tensors="pt",
        )
        single_chunk = test_encoding["input_ids"].shape[0] == 1

        if single_chunk:
            window = {k: v.to(self.device) for k, v in test_encoding.items()}
            # Add offset_mapping for single chunk
            enc_with_offsets = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=chunk_size,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            window["offset_mapping"] = enc_with_offsets["offset_mapping"].to(self.device)
            result = self._extract_answer_from_window(window, context, len(question))
            if result is None:
                return {"answer": "", "score": 0.0, "start": 0, "end": 0}
            if top_k == 1:
                return result
            else:
                return {"answers": [result]}

        # Sliding window for long contexts
        windows = self._encode_sliding_window(question, context, chunk_size, stride)
        all_answers = []
        for window in windows:
            result = self._extract_answer_from_window(window, context, len(question))
            if result is not None:
                all_answers.append(result)

        if not all_answers:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        if top_k == 1:
            best = find_best_answer(all_answers)
            return best
        else:
            sorted_answers = sorted(all_answers, key=lambda x: x["score"], reverse=True)[:top_k]
            return {"answers": sorted_answers}

    def batch_answer(
        self,
        examples: List[dict],
        chunk_size: int = 384,
        stride: int = 128,
    ) -> List[dict]:
        """Answer multiple question-context pairs.

        Args:
            examples: List of dicts with "question" and "context" keys.
            chunk_size: Max tokens per chunk.
            stride: Sliding window stride.

        Returns:
            List of answer dicts.
        """
        results = []
        for ex in examples:
            result = self.answer(
                question=ex["question"],
                context=ex["context"],
                chunk_size=chunk_size,
                stride=stride,
            )
            result["id"] = ex.get("id", None)
            result["question"] = ex["question"]
            results.append(result)
        return results

    def info(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "num_parameters": self.num_parameters,
            "is_micro": self.is_micro,
        }
