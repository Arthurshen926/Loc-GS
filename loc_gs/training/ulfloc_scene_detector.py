from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from loc_gs.stdloc_native.detector_target_refinement import build_lsf_detector_target


def pixel_yx_to_coarse_yx(pixel_yx: torch.Tensor | Any, *, stride: int = 8) -> torch.Tensor:
    points = torch.as_tensor(pixel_yx, dtype=torch.float32).reshape(-1, 2)
    scale = float(max(1, int(stride)))
    return ((points + 0.5) / scale) - 0.5


def rasterize_compact_detector_target(
    entry: Mapping[str, Any],
    *,
    coarse_height: int,
    coarse_width: int,
    stride: int = 8,
    sigma_cells: float = 1.0,
    solver_validity_power: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if "keypoint_yx" not in entry:
        raise KeyError("compact detector target entry must contain keypoint_yx")
    if "support_weights" not in entry:
        raise KeyError("compact detector target entry must contain support_weights")
    coarse_yx = pixel_yx_to_coarse_yx(entry["keypoint_yx"], stride=int(stride))
    support_weights = torch.as_tensor(entry["support_weights"], dtype=torch.float32).reshape(-1)
    solver_validity = torch.as_tensor(
        entry.get("solver_validity_weights", torch.ones_like(support_weights)),
        dtype=torch.float32,
    ).reshape(-1)
    if solver_validity.numel() != support_weights.numel():
        raise ValueError("solver_validity_weights must match support_weights")
    heatmap, weight, metadata = build_lsf_detector_target(
        projected_yx=coarse_yx,
        support_weights=support_weights,
        solver_validity_weights=solver_validity,
        solver_validity_power=float(solver_validity_power),
        height=int(coarse_height),
        width=int(coarse_width),
        sigma_px=float(sigma_cells),
    )
    negative_count = 0
    if "negative_keypoint_yx" in entry:
        negative_yx = pixel_yx_to_coarse_yx(entry["negative_keypoint_yx"], stride=int(stride))
        negative_weights = torch.as_tensor(entry.get("negative_weights", []), dtype=torch.float32).reshape(-1)
        if negative_yx.shape[0] != negative_weights.numel():
            raise ValueError("negative_weights must match negative_keypoint_yx rows")
        if negative_yx.shape[0] > 0:
            _negative_heatmap, negative_weight, negative_metadata = build_lsf_detector_target(
                projected_yx=negative_yx,
                support_weights=negative_weights,
                height=int(coarse_height),
                width=int(coarse_width),
                sigma_px=float(sigma_cells),
            )
            weight = torch.maximum(weight, negative_weight)
            negative_count = int(negative_metadata.get("point_count", negative_yx.shape[0]))
    metadata = dict(metadata)
    metadata.update(
        {
            "target_grid": "superpoint_coarse_descriptor",
            "descriptor_stride": int(stride),
            "coarse_height": int(coarse_height),
            "coarse_width": int(coarse_width),
            "negative_point_count": int(negative_count),
        }
    )
    return heatmap, weight, metadata


def scene_detector_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    positive_weight: float = 6.0,
    negative_weight: float = 0.25,
) -> torch.Tensor:
    pred = torch.as_tensor(prediction, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    tgt = torch.as_tensor(target, dtype=torch.float32).to(device=pred.device)
    w = torch.as_tensor(weight, dtype=torch.float32).to(device=pred.device)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} must match target {tuple(tgt.shape)}")
    if w.shape != tgt.shape:
        raise ValueError(f"weight shape {tuple(w.shape)} must match target {tuple(tgt.shape)}")
    pixel_weight = float(negative_weight) + w.clamp_min(0.0) + float(positive_weight) * tgt.clamp_min(0.0)
    loss = F.binary_cross_entropy(pred, tgt.clamp(0.0, 1.0), reduction="none")
    return (loss * pixel_weight).mean()


def make_trainable_detector_input(feature_map: torch.Tensor | Any) -> torch.Tensor:
    """Detach frozen ULF feature maps and remove PyTorch inference tensor state."""

    return torch.as_tensor(feature_map, dtype=torch.float32).detach().clone()


def load_detector_target_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"detector target artifact must be a dict: {path}")
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    split_audit = payload.get("split_audit", {})
    audit_split = str(split_audit.get("split_name", "")).strip() if isinstance(split_audit, Mapping) else ""
    if split.lower() == "test" or audit_split.lower() == "test":
        raise ValueError("refusing to load ULF-Loc detector targets from test split")
    if isinstance(split_audit, Mapping) and bool(split_audit.get("test_split_used", False)):
        raise ValueError("refusing to load ULF-Loc detector targets marked as test_split_used")
    if "targets" not in payload or not isinstance(payload["targets"], Mapping):
        raise KeyError("detector target artifact must contain a targets mapping")
    return payload
