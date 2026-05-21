from __future__ import annotations

from typing import Any

import torch

from loc_gs.diagnostics.alpha_composition import (
    alpha_composition_reliability,
    composition_support_aggregation,
)


def rendered_support_mask(
    *,
    contributor_ids: torch.Tensor | Any,
    contribution_weights: torch.Tensor | Any,
    gaussian_support: torch.Tensor | Any,
    depth_stability: torch.Tensor | Any | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a lightweight rendered ray/pixel support mask from LSF signals."""

    support = composition_support_aggregation(contributor_ids, contribution_weights, gaussian_support)
    alpha = alpha_composition_reliability(
        contribution_weights,
        depth_stability=depth_stability,
    )
    reliability = (support * alpha["reliability"]).clamp(0.0, 1.0)
    keep = reliability >= float(threshold)
    return {
        "support": support,
        "dominance": alpha["dominance"],
        "entropy": alpha["entropy"],
        "reliability": reliability,
        "keep": keep,
        "metadata": {
            "composition_proxy": True,
            "exact_raster_weights": False,
            "threshold": float(threshold),
        },
    }

