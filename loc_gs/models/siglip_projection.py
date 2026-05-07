"""Shared SigLIP2 projection utilities for grounding-aware supervision."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class SigLIP2FeatureProjection(nn.Module):
    """Project RADIO 1280d features into SigLIP2 visual embedding space.

    Uses the spatial feature projection from RADIO: 2 attention blocks + MLP.
    Maps to SigLIP2's *spatial* vision space (1536d).
    NOTE: For text grounding, use SigLIP2SummaryHead instead — it maps to
    the text-aligned summary space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(1280, num_heads=16, init_values=1e-5)
            for _ in range(2)
        ])
        self.mlp_fc1 = nn.Linear(1280, 1520)
        self.mlp_final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.blocks(x)
        x = self.mlp_fc1(x)
        x = self.mlp_final(x)
        return x

    @classmethod
    def from_extracted_weights(cls, ckpt_path: str) -> "SigLIP2FeatureProjection":
        """Load from already-extracted projection state dict (e.g. siglip2_feat_projection.pth)."""
        sd = torch.load(ckpt_path, map_location="cpu")
        proj = cls()
        proj.load_state_dict(sd, strict=True)
        return proj

    @classmethod
    def from_radio_checkpoint(cls, ckpt_path: str) -> "SigLIP2FeatureProjection":
        chk = torch.load(ckpt_path, map_location="cpu")
        sd = chk["state_dict"]
        proj = cls()
        proj_sd = {}
        prefix = "_feature_projections.siglip2-g."
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            new_k = k[len(prefix):]
            if new_k.startswith("mlp.fc1"):
                new_k = new_k.replace("mlp.fc1", "mlp_fc1")
            elif new_k.startswith("mlp.final"):
                new_k = new_k.replace("mlp.final", "mlp_final")
            proj_sd[new_k] = v.float()
        proj.load_state_dict(proj_sd, strict=True)
        return proj


class SigLIP2SummaryHead(nn.Module):
    """RADIO's SigLIP2 summary head — maps 1280d tokens to the text-aligned 1536d space.

    Architecture: Linear(1280→1520) + 2 residual blocks(LN+GELU+Linear) + final(LN+GELU+Linear→1536).
    This is ``_heads.siglip2-g`` from the RADIO checkpoint. Unlike the spatial feature projection,
    the summary head produces embeddings in the same space as SigLIP2 text embeddings, making it
    suitable for text grounding / open-vocabulary tasks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1280, 1520)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1520)),
            nn.Sequential(nn.LayerNorm(1520), nn.GELU(), nn.Linear(1520, 1520)),
        ])
        self.final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.fc1(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.final(x)

    @classmethod
    def from_extracted_weights(cls, ckpt_path: str) -> "SigLIP2SummaryHead":
        """Load from extracted state dict (e.g. siglip2_summary_head.pth)."""
        sd = torch.load(ckpt_path, map_location="cpu")
        head = cls()
        head.load_state_dict(sd, strict=True)
        return head

    @classmethod
    def from_radio_checkpoint(cls, ckpt_path: str) -> "SigLIP2SummaryHead":
        """Extract ``_heads.siglip2-g`` from a full RADIO checkpoint."""
        chk = torch.load(ckpt_path, map_location="cpu")
        sd = chk["state_dict"]
        head = cls()
        head_sd = {}
        prefix = "_heads.siglip2-g."
        for k, v in sd.items():
            if k.startswith(prefix):
                head_sd[k[len(prefix):]] = v.float()
        head.load_state_dict(head_sd, strict=True)
        return head
