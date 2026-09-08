# src/models/supervised_transformer.py
"""
Mel Spectrogram -> AudioTransformerInput -> Transformer Encoder
-> CLS Token (out[:, 0]) -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Logits
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .encoder import Encoder


class SupervisedTransformer(nn.Module):
    """Supervised Transformer Classifier for Bird Species ID."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        audio_cfg: Dict[str, Any] = config["audio"]
        model_cfg: Dict[str, Any] = config["model"]

        n_mels: int = audio_cfg["n_mels"]
        patch_size: int = model_cfg["patch_size"]
        embed_dim: int = model_cfg["embed_dim"]
        segment_size: int = audio_cfg["segment_size"]

        num_patches: int = segment_size // patch_size
        max_len: int = num_patches + 1 + 10  # +1 CLS + buffer

        self.encoder: Encoder = Encoder(
            n_mels=n_mels,
            patch_size=patch_size,
            embed_size=embed_dim,
            num_layers=model_cfg["num_layers"],
            heads=model_cfg["heads"],
            device=device,
            forward_expansion=model_cfg["forward_expansion"],
            dropout=model_cfg["dropout"],
            max_len=max_len,
        )

        self.head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(embed_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, time)
            mask: (B, 1, 1, seq_len) optional

        Returns:
            (B, num_classes) logits
        """
        enc_out: torch.Tensor = self.encoder(x, mask)
        cls_token: torch.Tensor = enc_out[:, 0]
        logits: torch.Tensor = self.head(cls_token)
        return logits

    def get_encoder(self) -> Encoder:
        return self.encoder
