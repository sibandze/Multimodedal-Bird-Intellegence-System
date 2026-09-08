# src/models/patch_embedding.py
"""
Input
-----
(B, 128, 600)

↓

Patch size = 25 frames

↓

24 patches

↓

Flatten

↓

(B, 24, 3200)

↓

Linear Projection

↓

(B, 24, 256)
"""

import torch
import torch.nn as nn


class SpectrogramPatchEmbedding(nn.Module):
    def __init__(
        self,
        n_mels: int,
        patch_size: int,
        embed_dim: int,
    ):  # (128, 25, 256)
        super().__init__()

        self.n_mels = n_mels
        self.patch_size = patch_size

        # each patch is flattened then projected
        self.projection = nn.Linear(n_mels * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_mels, time)
        returns: (B, num_patches, embed_dim)
        """
        B, M, T = x.shape
        assert M == self.n_mels, f"Expected {self.n_mels} Mel bins, got {M}."

        # ensure divisible
        T_trim = (T // self.patch_size) * self.patch_size
        x = x[:, :, :T_trim]

        # reshape into patches
        x = x.reshape(B, M, T_trim // self.patch_size, self.patch_size)

        # now: (B, M, num_patches, patch_size)

        # move patch dim next to mel
        x = x.permute(0, 2, 1, 3)
        # (B, num_patches, M, patch_size)
        x = x.contiguous()

        # flatten each patch
        x = x.reshape(B, -1, M * self.patch_size)
        # (B, num_patches, patch_dim)

        # linear projection
        x = self.projection(x)

        return x
