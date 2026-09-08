# src/models/transformer_block.py
from typing import Optional

import torch
import torch.nn as nn

from .self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_size: int, heads: int, dropout: float, forward_expansion: int
    ) -> None:
        super().__init__()
        self.attention: SelfAttention = SelfAttention(
            embed_size, heads, dropout=dropout
        )

        self.norm1: nn.LayerNorm = nn.LayerNorm(embed_size)
        self.norm2: nn.LayerNorm = nn.LayerNorm(embed_size)

        self.feed_forward: nn.Sequential = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention: torch.Tensor = self.attention(value, key, query, mask)

        x: torch.Tensor = self.dropout(self.norm1(attention + query))

        forward: torch.Tensor = self.feed_forward(x)

        out: torch.Tensor = self.dropout(self.norm2(forward + x))

        return out
