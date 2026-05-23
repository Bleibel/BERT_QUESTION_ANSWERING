"""Custom Micro-BERT configuration (~1.8M parameters).

This module defines a drastically scaled-down BERT architecture suitable
for educational purposes and lightweight deployments.
"""

from transformers import BertConfig


def get_micro_bert_config() -> BertConfig:
    """Return a ~1.8M-parameter BERT configuration.

    Architecture:
        - 2 transformer layers
        - 56 hidden dimensions
        - 2 attention heads
        - 128 intermediate (FFN) dimensions
        - 512 max position embeddings
        - ~30522 WordPiece vocabulary

    Total parameters: ~1,795,642
    """
    return BertConfig(
        vocab_size=30522,
        hidden_size=56,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
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
