import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V. mask: broadcastable to scores."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


class MultiHeadAttention(nn.Module):
    """H heads; mỗi head chiều d_k = d_model // num_heads. Cho phép Q,K,V khác nhau (cross-attention)."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model) -> (B, H, L, d_k)
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, d_k) -> (B, L, d_model)
        b, h, l, dk = x.shape
        x = x.transpose(1, 2).contiguous().view(b, l, h * dk)
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(q))
        K = self._split_heads(self.W_k(k))
        V = self._split_heads(self.W_v(v))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self._merge_heads(out))
