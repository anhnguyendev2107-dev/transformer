"""python -m pytest tests/test_shapes.py -q"""

import torch

from transformer import Transformer


def test_transformer_forward_shapes():
    pad = 0
    vocab = 32
    B, src_l, tgt_l = 2, 5, 7
    model = Transformer(
        vocab_size=vocab,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        pad_idx=pad,
    )
    src = torch.randint(3, vocab, (B, src_l))
    src[0, 3:] = pad
    tgt = torch.randint(1, 3, (B, tgt_l))
    logits = model(src, tgt)
    assert logits.shape == (B, tgt_l, vocab)


def test_multi_head_attention_mask():
    from transformer.attention import MultiHeadAttention

    m = MultiHeadAttention(64, 8)
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 1, 1, 10)
    mask[0, :, :, 5:] = 0
    y = m(x, x, x, mask)
    assert y.shape == (2, 10, 64)
