# src/models/encoder.py
from typing import Optional

import torch
import torch.nn as nn

from .audio_transformer_input import AudioTransformerInput
from .transformer_block import TransformerBlock


class Encoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        patch_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        device: torch.device | str,
        forward_expansion: int,
        dropout: float,
        max_len: int = 1000,
    ) -> None:
        super().__init__()
        self.embed_size: int = embed_size
        self.device: torch.device = torch.device(device)

        self.input_layer: AudioTransformerInput = AudioTransformerInput(
            n_mels=n_mels,
            patch_size=patch_size,
            embed_dim=embed_size,
            max_len=max_len,
            dropout=dropout,
        )

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, time)
            mask: attention mask (B, 1, 1, num_patches) or None

        Returns:
            (B, num_patches, embed_size)
        """
        out: torch.Tensor = self.dropout(self.input_layer(x))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

    def get_output_dim(self) -> int:
        return self.embed_size
