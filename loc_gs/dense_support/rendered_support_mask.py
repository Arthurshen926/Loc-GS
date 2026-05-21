from __future__ import annotations

from typing import Any

import torch

from loc_gs.diagnostics.rendered_reliability import rendered_support_mask


def build_dense_residual_support(
    *,
    contributor_ids: torch.Tensor | Any,
    contribution_weights: torch.Tensor | Any,
    gaussian_support: torch.Tensor | Any,
    pose_leverage: torch.Tensor | Any | None = None,
    depth_stability: torch.Tensor | Any | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Build the v1 proxy dense residual support mask."""

    base = rendered_support_mask(
        contributor_ids=contributor_ids,
        contribution_weights=contribution_weights,
        gaussian_support=gaussian_support,
        depth_stability=depth_stability,
        threshold=0.0,
    )
    reliability = base["reliability"].float().cpu()
    if pose_leverage is not None:
        leverage = torch.as_tensor(pose_leverage, dtype=torch.float32).reshape(reliability.shape).cpu().clamp(0.0, 1.0)
        reliability = reliability * leverage
    else:
        leverage = torch.ones_like(reliability)
    keep = reliability >= float(threshold)
    return {
        **base,
        "pose_leverage": leverage,
        "residual_support": reliability.clamp(0.0, 1.0),
        "keep": keep,
        "metadata": {
            **base["metadata"],
            "dense_support_version": "v1_proxy",
            "threshold": float(threshold),
        },
    }

