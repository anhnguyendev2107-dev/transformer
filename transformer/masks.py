import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """seq: (B, L) token ids. Trả về (B, 1, 1, L) — 1 giữ, 0 tại vị trí pad (để mask attention)."""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(dtype=torch.float)


def create_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Ma trận tam giác dưới (B, 1, L, L): token i chỉ nhìn j <= i."""
    return torch.tril(torch.ones(size, size, device=device)).view(1, 1, size, size)
