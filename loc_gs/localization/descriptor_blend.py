from __future__ import annotations

import torch
import torch.nn.functional as F


def gated_residual_descriptor_blend(
    ply_desc: torch.Tensor,
    hybrid_desc: torch.Tensor,
    gate: torch.Tensor | None = None,
    alpha_max: float = 0.05,
) -> torch.Tensor:
    """Blend a small learned residual into a protected PLY/STDLoc descriptor.

    The current hybrid head predicts a full descriptor, so the residual is
    interpreted as the difference between the hybrid descriptor and the PLY
    descriptor.  alpha_max keeps the default path close to the STDLoc bank.
    """

    ply = F.normalize(ply_desc.float(), p=2, dim=-1 if ply_desc.dim() == 2 else 1)
    hybrid = F.normalize(hybrid_desc.float(), p=2, dim=-1 if hybrid_desc.dim() == 2 else 1)
    alpha = float(max(alpha_max, 0.0))
    if alpha == 0.0:
        return ply
    if gate is None:
        gate_value = torch.ones_like(ply[..., :1]) if ply.dim() == 2 else torch.ones_like(ply[:, :1])
    else:
        gate_value = gate.to(device=ply.device, dtype=ply.dtype).clamp(0.0, 1.0)
        if ply.dim() == 2:
            gate_value = gate_value.reshape(-1, 1)
        elif gate_value.dim() == 3:
            gate_value = gate_value.unsqueeze(1)
    return F.normalize(ply + alpha * gate_value * (hybrid - ply), p=2, dim=-1 if ply.dim() == 2 else 1)
