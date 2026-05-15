from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class UnifiedLFFDescriptor(nn.Module):
    """Export-aligned bounded residual descriptor for Unified LFF-v2.

    The STDLoc/native descriptor bank is kept as a frozen trust-region anchor.
    Training learns a residual vector and a per-landmark gate that may either
    apply or suppress the residual.  This module is meant to be used in the
    training forward path, not only during post-hoc export.
    """

    def __init__(
        self,
        base_descriptors: torch.Tensor,
        *,
        residual_init: torch.Tensor | None = None,
        gate_logit_init: torch.Tensor | None = None,
        alpha_max: float = 0.05,
        init_gate: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if base_descriptors.dim() != 2:
            raise ValueError("base_descriptors must have shape [num_landmarks, descriptor_dim]")
        if base_descriptors.numel() == 0:
            raise ValueError("base_descriptors must be non-empty")
        base = F.normalize(base_descriptors.float(), p=2, dim=-1, eps=float(eps))
        self.register_buffer("base_descriptors", base)
        self.alpha_max = float(max(alpha_max, 0.0))
        self.eps = float(eps)
        if residual_init is None:
            residual = torch.zeros_like(base)
        else:
            residual = residual_init.float()
            if residual.shape != base.shape:
                raise ValueError(
                    f"residual_init has shape {tuple(residual.shape)}, expected {tuple(base.shape)}"
                )
        self.residual = nn.Parameter(residual.clone())
        if gate_logit_init is None:
            gate_logit = torch.full((base.shape[0],), _logit(init_gate), dtype=torch.float32)
        else:
            gate_logit = gate_logit_init.float().reshape(-1)
            if gate_logit.shape[0] != base.shape[0]:
                raise ValueError(
                    f"gate_logit_init has {gate_logit.shape[0]} rows, expected {base.shape[0]}"
                )
        self.gate_logit = nn.Parameter(gate_logit.clone())

    @property
    def num_landmarks(self) -> int:
        return int(self.base_descriptors.shape[0])

    @property
    def descriptor_dim(self) -> int:
        return int(self.base_descriptors.shape[1])

    def _ids(self, ids: torch.Tensor | None) -> torch.Tensor | slice:
        if ids is None:
            return slice(None)
        index = ids.to(device=self.base_descriptors.device, dtype=torch.long).reshape(-1)
        if index.numel() and (int(index.min().item()) < 0 or int(index.max().item()) >= self.num_landmarks):
            raise IndexError("descriptor ids are out of bounds")
        return index

    def gate(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit[self._ids(ids)])

    def alpha(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.alpha_max * self.gate(ids)

    def bounded_residual(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.residual[self._ids(ids)]
        norm = raw.norm(dim=-1, keepdim=True)
        return raw / norm.clamp_min(1.0)

    def forward(self, ids: torch.Tensor | None = None) -> torch.Tensor:
        index = self._ids(ids)
        base = self.base_descriptors[index]
        residual = self.bounded_residual(ids)
        alpha = self.alpha(ids).reshape(-1, 1)
        return F.normalize(base + alpha * residual, p=2, dim=-1, eps=self.eps)

    def trust_region_loss(
        self,
        ids: torch.Tensor | None = None,
        *,
        l1_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        index = self._ids(ids)
        residual = self.bounded_residual(ids)
        alpha = self.alpha(ids).reshape(-1, 1)
        weighted_residual = alpha * residual
        residual_l2 = weighted_residual.square().sum(dim=-1).mean()
        alpha_l1 = alpha.reshape(-1).abs().mean()
        loss = residual_l2 + float(l1_weight) * alpha_l1
        return {
            "loss": loss,
            "residual_l2": residual_l2.detach(),
            "alpha_l1": alpha_l1.detach(),
        }
