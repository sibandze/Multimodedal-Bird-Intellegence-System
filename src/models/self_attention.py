# src/models/self_attention.py
from typing import Optional
import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_size % heads != 0:
            raise ValueError(
                f"embed_size {embed_size} must be divisible by heads {heads}"
            )

        self.embed_size: int = embed_size
        self.heads: int = heads
        self.head_dim: int = embed_size // heads

        self.values: nn.Linear = nn.Linear(embed_size, embed_size, bias=False)
        self.keys: nn.Linear = nn.Linear(embed_size, embed_size, bias=False)
        self.queries: nn.Linear = nn.Linear(embed_size, embed_size, bias=False)

        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.fc_out: nn.Linear = nn.Linear(embed_size, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            values, keys, query: [N, seq_len, embed_size]
            mask: [N, 1, 1, seq_len] or [N, 1, seq_len, seq_len]
        Returns:
            out: [N, seq_len, embed_size]
        """
        n, seq_len, _ = query.shape

        v: torch.Tensor = self.values(values)
        k: torch.Tensor = self.keys(keys)
        q: torch.Tensor = self.queries(query)

        v = v.view(n, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(n, seq_len, self.heads, self.head_dim).transpose(1, 2)
        q = q.view(n, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # TODO: replace with F.scaled_dot_product_attention once bottleneck confirmed
        energy: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e10"))

        attention: torch.Tensor = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        out: torch.Tensor = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(n, seq_len, self.embed_size)
        out = self.fc_out(out)

        return out
