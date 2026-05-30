"""Custom Micro-BERT configuration (~10M parameters).

This module defines a scaled-down BERT architecture with ~10M parameters,
a ~5x increase from the original 1.8M version, for stronger QA performance
while remaining lightweight.
"""

from transformers import BertConfig


def get_micro_bert_config() -> BertConfig:
    """Return a ~10M-parameter BERT configuration.

    Architecture:
        - 4 transformer layers
        - 256 hidden dimensions
        - 4 attention heads
        - 512 intermediate (FFN) dimensions
        - 512 max position embeddings
        - ~30522 WordPiece vocabulary

    Total parameters: ~10,120,450
    """
    return BertConfig(
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
    )


def count_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
