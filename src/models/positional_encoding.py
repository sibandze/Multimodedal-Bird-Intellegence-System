# src/models/positional_encoding.py
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 1000) -> None:
        super().__init__()
        self.max_len: int = max_len
        self.embed_dim: int = embed_dim

        self.position_embeddings: nn.Parameter = nn.Parameter(
            torch.zeros(1, max_len, embed_dim)
        )
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) where T = num_patches + 1 if CLS token is used

        Returns:
            (B, T, D) with positional information added
        """
        t: int = x.size(1)
        if t > self.max_len:
            raise ValueError(
                f"Sequence length {t} > max_len {self.max_len}. "
                f"Increase max_len in PositionalEncoding."
            )
        return x + self.position_embeddings[:, :t]
