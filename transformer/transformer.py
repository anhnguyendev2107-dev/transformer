import torch
import torch.nn as nn

from .decoder_layer import DecoderLayer
from .encoder_layer import EncoderLayer
from .masks import create_causal_mask, create_padding_mask
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """Encoder–Decoder Transformer (Vaswani et al., 2017)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.encoder = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B, src_len)
        src_mask = create_padding_mask(src, self.pad_idx)
        x = self.pos_enc(self.src_embed(src) * (self.d_model**0.5))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        # tgt: (B, tgt_len); memory: encoder output
        tgt_pad = create_padding_mask(tgt, self.pad_idx)
        causal = create_causal_mask(tgt.size(1), tgt.device)
        tgt_mask = tgt_pad * causal

        src_key_mask = create_padding_mask(src, self.pad_idx)

        x = self.pos_enc(self.tgt_embed(tgt) * (self.d_model**0.5))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_key_mask)
        return self.fc_out(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src)
        return self.decode(tgt, memory, src)
