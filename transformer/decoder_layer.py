import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Masked self-attention + cross-attention (encoder) + FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        y = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(y))
        y = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(y))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x
