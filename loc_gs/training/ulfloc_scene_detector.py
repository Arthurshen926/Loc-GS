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


def scale_pixel_yx(
    pixel_yx: torch.Tensor | Any,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    points = torch.as_tensor(pixel_yx, dtype=torch.float32).reshape(-1, 2)
    if int(source_height) <= 0 or int(source_width) <= 0:
        raise ValueError("source_height/source_width must be positive")
    if int(target_height) <= 0 or int(target_width) <= 0:
        raise ValueError("target_height/target_width must be positive")
    scale_y = float(target_height) / float(source_height)
    scale_x = float(target_width) / float(source_width)
    out = points.clone()
    if out.numel() > 0:
        out[:, 0] = out[:, 0] * scale_y
        out[:, 1] = out[:, 1] * scale_x
    return out


def _entry_source_size(entry: Mapping[str, Any], *, fallback_height: int, fallback_width: int) -> tuple[int, int]:
    source_height = int(entry.get("height", entry.get("source_height", fallback_height)))
    source_width = int(entry.get("width", entry.get("source_width", fallback_width)))
    return source_height, source_width


def _rasterize_stdloc_fullres_pixels(
    pixel_yx: torch.Tensor | Any,
    *,
    height: int,
    width: int,
    weights: torch.Tensor | Any | None = None,
    binary: bool = True,
    sigma_px: float = 0.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    points = torch.as_tensor(pixel_yx, dtype=torch.float32).reshape(-1, 2)
    h = int(height)
    w = int(width)
    target = torch.zeros((1, 1, h, w), dtype=torch.float32)
    if points.numel() == 0:
        return target, {"point_count": 0, "in_bounds_count": 0}
    values = torch.ones(points.shape[0], dtype=torch.float32)
    if not bool(binary) and weights is not None:
        raw_weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        if raw_weights.numel() == points.shape[0]:
            values = raw_weights.clamp(0.0, 1.0)
    coords = points.long()
    y = coords[:, 0]
    x = coords[:, 1]
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if bool(in_bounds.any()):
        yy = y[in_bounds]
        xx = x[in_bounds]
        vv = values[in_bounds]
        flat = target.reshape(-1)
        sigma = float(sigma_px)
        if sigma <= 0.0:
            linear = yy * w + xx
            flat.scatter_reduce_(0, linear, vv, reduce="amax", include_self=True)
        else:
            radius = max(1, int(round(3.0 * sigma)))
            offset_y, offset_x = torch.meshgrid(
                torch.arange(-radius, radius + 1, dtype=torch.long),
                torch.arange(-radius, radius + 1, dtype=torch.long),
                indexing="ij",
            )
            offsets_y = offset_y.reshape(1, -1)
            offsets_x = offset_x.reshape(1, -1)
            patch_y = yy.reshape(-1, 1) + offsets_y
            patch_x = xx.reshape(-1, 1) + offsets_x
            patch_valid = (patch_y >= 0) & (patch_y < h) & (patch_x >= 0) & (patch_x < w)
            dist2 = (offsets_y.float() ** 2) + (offsets_x.float() ** 2)
            kernel = torch.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12)).reshape(1, -1)
            patch_values = (vv.reshape(-1, 1) * kernel).clamp(0.0, 1.0)
            if bool(patch_valid.any()):
                linear = (patch_y * w + patch_x)[patch_valid]
                flat.scatter_reduce_(0, linear, patch_values[patch_valid], reduce="amax", include_self=True)
    return target, {"point_count": int(points.shape[0]), "in_bounds_count": int(in_bounds.sum().item())}


def rasterize_fullres_scene_detector_components(
    entry: Mapping[str, Any],
    *,
    target_height: int,
    target_width: int,
    sigma_px: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Rasterize STDLoc-style landmark projection targets on full-res feature maps."""

    source_height, source_width = _entry_source_size(
        entry,
        fallback_height=int(target_height),
        fallback_width=int(target_width),
    )
    base_yx = scale_pixel_yx(
        entry.get("keypoint_yx", []),
        source_height=source_height,
        source_width=source_width,
        target_height=int(target_height),
        target_width=int(target_width),
    )
    visibility_point_count = int(base_yx.shape[0])
    sp_teacher_yx = scale_pixel_yx(
        entry.get("sp_teacher_keypoint_yx", []),
        source_height=source_height,
        source_width=source_width,
        target_height=int(target_height),
        target_width=int(target_width),
    )
    sp_teacher_point_count = int(sp_teacher_yx.shape[0])
    if sp_teacher_point_count > 0:
        base_yx = torch.cat((base_yx, sp_teacher_yx), dim=0)
    base, metadata = _rasterize_stdloc_fullres_pixels(
        base_yx,
        height=int(target_height),
        width=int(target_width),
        binary=True,
        sigma_px=float(sigma_px),
    )

    solver_pos_yx = scale_pixel_yx(
        entry.get("solver_positive_keypoint_yx", []),
        source_height=source_height,
        source_width=source_width,
        target_height=int(target_height),
        target_width=int(target_width),
    )
    solver_pos_weights = torch.as_tensor(entry.get("solver_positive_weights", []), dtype=torch.float32).reshape(-1)
    if solver_pos_weights.numel() != solver_pos_yx.shape[0]:
        solver_pos_weights = torch.ones(solver_pos_yx.shape[0], dtype=torch.float32)
    solver_positive, solver_metadata = _rasterize_stdloc_fullres_pixels(
        solver_pos_yx,
        height=int(target_height),
        width=int(target_width),
        weights=solver_pos_weights,
        binary=False,
        sigma_px=float(sigma_px),
    )

    negative_yx = scale_pixel_yx(
        entry.get("negative_keypoint_yx", []),
        source_height=source_height,
        source_width=source_width,
        target_height=int(target_height),
        target_width=int(target_width),
    )
    negative_weights = torch.as_tensor(entry.get("negative_weights", []), dtype=torch.float32).reshape(-1)
    if negative_weights.numel() != negative_yx.shape[0]:
        negative_weights = torch.ones(negative_yx.shape[0], dtype=torch.float32)
    suppression, negative_metadata = _rasterize_stdloc_fullres_pixels(
        negative_yx,
        height=int(target_height),
        width=int(target_width),
        weights=negative_weights,
        binary=False,
        sigma_px=float(sigma_px),
    )

    metadata = dict(metadata)
    metadata.update(
        {
            "target_style": "stdloc_binary_projection",
            "target_grid": "stdloc_full_resolution_feature_map",
            "source_height": int(source_height),
            "source_width": int(source_width),
            "target_height": int(target_height),
            "target_width": int(target_width),
            "visibility_point_count": int(visibility_point_count),
            "sp_teacher_point_count": int(sp_teacher_point_count),
            "solver_positive_point_count": int(solver_metadata.get("point_count", solver_pos_yx.shape[0])),
            "solver_positive_in_bounds_count": int(solver_metadata.get("in_bounds_count", 0)),
            "negative_point_count": int(negative_metadata.get("point_count", negative_yx.shape[0])),
            "negative_in_bounds_count": int(negative_metadata.get("in_bounds_count", 0)),
            "sigma_px": float(sigma_px),
        }
    )
    return base, solver_positive, suppression, metadata


def rasterize_compact_detector_target(
    entry: Mapping[str, Any],
    *,
    coarse_height: int,
    coarse_width: int,
    stride: int = 8,
    sigma_cells: float = 1.0,
    solver_validity_power: float = 0.0,
    return_suppression: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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
    suppression = torch.zeros_like(heatmap)
    if "negative_keypoint_yx" in entry:
        negative_yx = pixel_yx_to_coarse_yx(entry["negative_keypoint_yx"], stride=int(stride))
        negative_weights = torch.as_tensor(entry.get("negative_weights", []), dtype=torch.float32).reshape(-1)
        if negative_yx.shape[0] != negative_weights.numel():
            raise ValueError("negative_weights must match negative_keypoint_yx rows")
        if negative_yx.shape[0] > 0:
            negative_heatmap, negative_weight, negative_metadata = build_lsf_detector_target(
                projected_yx=negative_yx,
                support_weights=negative_weights,
                height=int(coarse_height),
                width=int(coarse_width),
                sigma_px=float(sigma_cells),
            )
            suppression = negative_heatmap
            if not bool(return_suppression):
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
    if bool(return_suppression):
        return heatmap, weight, suppression, metadata
    return heatmap, weight, metadata


def _rasterize_optional_points(
    entry: Mapping[str, Any],
    *,
    point_key: str,
    weight_key: str,
    coarse_height: int,
    coarse_width: int,
    stride: int,
    sigma_cells: float,
) -> torch.Tensor:
    if point_key not in entry:
        return torch.zeros((1, 1, int(coarse_height), int(coarse_width)), dtype=torch.float32)
    points = pixel_yx_to_coarse_yx(entry[point_key], stride=int(stride))
    weights = torch.as_tensor(entry.get(weight_key, []), dtype=torch.float32).reshape(-1)
    if points.shape[0] != weights.numel():
        raise ValueError(f"{weight_key} must match {point_key} rows")
    if points.shape[0] == 0:
        return torch.zeros((1, 1, int(coarse_height), int(coarse_width)), dtype=torch.float32)
    heatmap, _weight, _metadata = build_lsf_detector_target(
        projected_yx=points,
        support_weights=weights,
        height=int(coarse_height),
        width=int(coarse_width),
        sigma_px=float(sigma_cells),
    )
    return heatmap


def rasterize_scene_detector_components(
    entry: Mapping[str, Any],
    *,
    coarse_height: int,
    coarse_width: int,
    stride: int = 8,
    sigma_cells: float = 1.0,
    solver_validity_power: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    visibility, _visibility_weight, suppression, metadata = rasterize_compact_detector_target(
        entry,
        coarse_height=int(coarse_height),
        coarse_width=int(coarse_width),
        stride=int(stride),
        sigma_cells=float(sigma_cells),
        solver_validity_power=float(solver_validity_power),
        return_suppression=True,
    )
    has_sp_teacher = "sp_teacher_keypoint_yx" in entry and torch.as_tensor(
        entry.get("sp_teacher_keypoint_yx", []), dtype=torch.float32
    ).numel() > 0
    if has_sp_teacher:
        teacher = _rasterize_optional_points(
            entry,
            point_key="sp_teacher_keypoint_yx",
            weight_key="sp_teacher_weights",
            coarse_height=int(coarse_height),
            coarse_width=int(coarse_width),
            stride=int(stride),
            sigma_cells=float(sigma_cells),
        )
        sp_teacher_source = "superpoint_teacher"
    else:
        teacher = visibility.clone()
        sp_teacher_source = "visibility_fallback"
    solver_positive = _rasterize_optional_points(
        entry,
        point_key="solver_positive_keypoint_yx",
        weight_key="solver_positive_weights",
        coarse_height=int(coarse_height),
        coarse_width=int(coarse_width),
        stride=int(stride),
        sigma_cells=float(sigma_cells),
    )
    metadata = dict(metadata)
    metadata["visibility_point_count"] = int(metadata.get("point_count", 0))
    metadata["sp_teacher_point_count"] = int(
        torch.as_tensor(entry.get("sp_teacher_keypoint_yx", []), dtype=torch.float32).reshape(-1, 2).shape[0]
    )
    metadata["sp_teacher_source"] = sp_teacher_source
    metadata["solver_positive_point_count"] = int(
        torch.as_tensor(entry.get("solver_positive_keypoint_yx", []), dtype=torch.float32).reshape(-1, 2).shape[0]
    )
    return teacher, visibility, solver_positive, suppression, metadata


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


def scene_detector_loss_with_suppression(
    prediction: torch.Tensor,
    positive_target: torch.Tensor,
    positive_weight: torch.Tensor,
    suppression_target: torch.Tensor,
    *,
    positive_weight_scale: float = 6.0,
    suppression_weight_scale: float = 2.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    pred = torch.as_tensor(prediction, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    pos = torch.as_tensor(positive_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    pos_w = torch.as_tensor(positive_weight, dtype=torch.float32).to(device=pred.device).clamp_min(0.0)
    suppress = torch.as_tensor(suppression_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    if pred.shape != pos.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} must match positive target {tuple(pos.shape)}")
    if pos_w.shape != pos.shape:
        raise ValueError(f"positive weight shape {tuple(pos_w.shape)} must match positive target {tuple(pos.shape)}")
    if suppress.shape != pos.shape:
        raise ValueError(f"suppression target shape {tuple(suppress.shape)} must match positive target {tuple(pos.shape)}")
    bce_pos = F.binary_cross_entropy(pred, pos, reduction="none")
    bce_suppress = F.binary_cross_entropy(pred, torch.zeros_like(pred), reduction="none")
    loss = bce_pos * (float(background_weight) + float(positive_weight_scale) * pos + pos_w)
    loss = loss + float(suppression_weight_scale) * suppress * bce_suppress
    return loss.mean()


def compose_teacher_preserving_detector_targets(
    sp_teacher_target: torch.Tensor,
    visibility_target: torch.Tensor,
    solver_positive_target: torch.Tensor,
    solver_negative_target: torch.Tensor,
    *,
    residual_alpha: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge visibility/solver detector feedback without replacing SuperPoint teacher.

    The direct scene detector is only paper-safe as a residual over the native
    query detector.  Visibility and solver-positive points outside the
    SuperPoint teacher are therefore capped by ``residual_alpha``; solver
    negatives are ignored where they overlap teacher positives.
    """

    teacher = torch.as_tensor(sp_teacher_target, dtype=torch.float32).clamp(0.0, 1.0)
    visibility = torch.as_tensor(visibility_target, dtype=torch.float32).to(device=teacher.device).clamp(0.0, 1.0)
    solver_pos = torch.as_tensor(solver_positive_target, dtype=torch.float32).to(device=teacher.device).clamp(0.0, 1.0)
    solver_neg = torch.as_tensor(solver_negative_target, dtype=torch.float32).to(device=teacher.device).clamp(0.0, 1.0)
    for name, target in (
        ("visibility_target", visibility),
        ("solver_positive_target", solver_pos),
        ("solver_negative_target", solver_neg),
    ):
        if target.shape != teacher.shape:
            raise ValueError(f"{name} shape {tuple(target.shape)} must match teacher shape {tuple(teacher.shape)}")
    alpha = max(0.0, min(1.0, float(residual_alpha)))
    residual_positive = torch.maximum(visibility, solver_pos) * (1.0 - teacher)
    positive = torch.maximum(teacher, residual_positive * alpha).clamp(0.0, 1.0)
    suppression = solver_neg * (1.0 - teacher)
    return positive, suppression.clamp(0.0, 1.0)


def build_scene_detector_loss_terms(
    prediction: torch.Tensor,
    *,
    sp_teacher_target: torch.Tensor,
    visibility_target: torch.Tensor,
    solver_positive_target: torch.Tensor,
    solver_negative_target: torch.Tensor,
    sp_teacher_weight: float = 1.0,
    visibility_weight: float = 1.0,
    solver_positive_weight: float = 1.0,
    solver_negative_weight: float = 1.0,
    residual_alpha: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Build explicit detector loss components for direct heatmap training."""

    pred = torch.as_tensor(prediction, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    teacher = torch.as_tensor(sp_teacher_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    visibility = torch.as_tensor(visibility_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    solver_pos = torch.as_tensor(solver_positive_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    solver_neg = torch.as_tensor(solver_negative_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    for name, target in (
        ("sp_teacher_target", teacher),
        ("visibility_target", visibility),
        ("solver_positive_target", solver_pos),
        ("solver_negative_target", solver_neg),
    ):
        if target.shape != pred.shape:
            raise ValueError(f"{name} shape {tuple(target.shape)} must match prediction {tuple(pred.shape)}")
    residual_positive, residual_negative = compose_teacher_preserving_detector_targets(
        teacher,
        visibility,
        solver_pos,
        solver_neg,
        residual_alpha=float(residual_alpha),
    )
    sp_teacher_loss = F.binary_cross_entropy(pred, teacher, reduction="mean") * float(sp_teacher_weight)
    visibility_loss = F.binary_cross_entropy(pred, residual_positive, reduction="mean") * float(visibility_weight)
    solver_positive_residual_loss = (
        F.binary_cross_entropy(pred, residual_positive, reduction="none") * residual_positive
    ).mean() * float(solver_positive_weight)
    solver_negative_suppression_loss = (
        F.binary_cross_entropy(pred, torch.zeros_like(pred), reduction="none") * residual_negative
    ).mean() * float(solver_negative_weight)
    total = sp_teacher_loss + visibility_loss + solver_positive_residual_loss + solver_negative_suppression_loss
    return {
        "sp_teacher_loss": sp_teacher_loss,
        "visibility_loss": visibility_loss,
        "solver_positive_residual_loss": solver_positive_residual_loss,
        "solver_negative_suppression_loss": solver_negative_suppression_loss,
        "total_loss": total,
    }


def build_stdloc_fullres_detector_loss_terms(
    prediction: torch.Tensor,
    *,
    base_target: torch.Tensor,
    solver_positive_target: torch.Tensor,
    solver_negative_target: torch.Tensor,
    residual_alpha: float = 0.0,
    solver_negative_weight: float = 1.0,
    background_weight: float = 1.0,
    positive_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """STDLoc-style full-resolution detector BCE with bounded solver residuals.

    With ``residual_alpha=0`` this is exactly the STDLoc landmark-projection
    BCE target.  Solver feedback is only allowed to add a small residual
    positive on non-base pixels and a suppression loss on non-base negatives.
    """

    pred = torch.as_tensor(prediction, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    base = torch.as_tensor(base_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    solver_pos = torch.as_tensor(solver_positive_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    solver_neg = torch.as_tensor(solver_negative_target, dtype=torch.float32).to(device=pred.device).clamp(0.0, 1.0)
    for name, target in (
        ("base_target", base),
        ("solver_positive_target", solver_pos),
        ("solver_negative_target", solver_neg),
    ):
        if target.shape != pred.shape:
            raise ValueError(f"{name} shape {tuple(target.shape)} must match prediction {tuple(pred.shape)}")

    alpha = max(0.0, min(1.0, float(residual_alpha)))
    base_bce = F.binary_cross_entropy(pred, base, reduction="none")
    base_weight = float(background_weight) + float(positive_weight) * base
    stdloc_bce_loss = (base_bce * base_weight).mean()
    residual_mask = (1.0 - base).clamp(0.0, 1.0)
    solver_positive_residual = (solver_pos * residual_mask).clamp(0.0, 1.0)
    solver_negative_residual = (solver_neg * residual_mask).clamp(0.0, 1.0)

    positive_target = (solver_positive_residual * alpha).clamp(0.0, 1.0)
    solver_positive_residual_loss = (
        F.binary_cross_entropy(pred, positive_target, reduction="none") * solver_positive_residual * alpha
    ).mean()
    solver_negative_suppression_loss = (
        F.binary_cross_entropy(pred, torch.zeros_like(pred), reduction="none")
        * solver_negative_residual
        * alpha
        * float(solver_negative_weight)
    ).mean()
    total = stdloc_bce_loss + solver_positive_residual_loss + solver_negative_suppression_loss
    return {
        "stdloc_bce_loss": stdloc_bce_loss,
        "solver_positive_residual_loss": solver_positive_residual_loss,
        "solver_negative_suppression_loss": solver_negative_suppression_loss,
        "total_loss": total,
    }


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
    def is_test_alias(value: str) -> bool:
        lowered = str(value).strip().lower()
        return lowered == "test" or lowered == "official_test" or lowered.endswith("_test")

    if is_test_alias(split) or is_test_alias(audit_split):
        raise ValueError("refusing to load ULF-Loc detector targets from test split")
    if isinstance(split_audit, Mapping) and (
        bool(split_audit.get("test_split_used", False)) or bool(split_audit.get("official_test_used", False))
    ):
        raise ValueError("refusing to load ULF-Loc detector targets marked as test_split_used")
    if "targets" not in payload or not isinstance(payload["targets"], Mapping):
        raise KeyError("detector target artifact must contain a targets mapping")
    return payload
