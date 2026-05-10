import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Một tầng encoder: self-attention + FFN, residual + LayerNorm (theo Vaswani et al.)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        # src_mask: broadcast tới (B, H, L, L) — thường (B,1,1,L) cho key padding
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
