from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from loc_gs.feedback.ray_attributed_solver_feedback import normalize_contributors, validate_split_name


SCHEMA_VERSION = "rendered_feedback_augmentation_v1"
SUPPORTED_MODES = {"observed_only", "rendered_depth_augmented"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _float(value, default)))


def validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_MODES:
        supported = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"mode must be one of {supported}, got {mode!r}")
    return normalized


def _record_split(record: Mapping[str, Any], default: str) -> str:
    split = str(record.get("split_name", record.get("split", default))).strip() or default
    return validate_split_name(split)


def _contributors(record: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    contributors = record.get("ray_contributors", record.get("contributors", []))
    if not isinstance(contributors, Sequence) or isinstance(contributors, (str, bytes)):
        raise TypeError("observation contributors must be a sequence")
    return contributors


def _observation_quality(record: Mapping[str, Any], reprojection_threshold_px: float) -> float:
    descriptor = _clamp01(record.get("descriptor_score"), 1.0)
    local_geometry = _clamp01(record.get("local_geometry_score"), 1.0)
    reproj = _float(record.get("reprojection_error_px"), 0.0)
    reproj_quality = max(0.0, min(1.0, 1.0 - reproj / max(float(reprojection_threshold_px), 1e-6)))
    return float(descriptor * local_geometry * reproj_quality)


def _artifact_score(record: Mapping[str, Any], *, depth_margin: float) -> float:
    explicit = _clamp01(record.get("artifact_score"), 0.0)
    expected = _float(record.get("expected_depth"), 0.0)
    rendered = _float(record.get("rendered_depth"), 0.0)
    if expected <= 0.0 or rendered <= 0.0 or rendered >= expected - float(depth_margin):
        return explicit
    depth_gap = max(0.0, min(1.0, (expected - rendered) / max(expected, 1e-6)))
    return max(explicit, depth_gap)


def _artifact_factor(
    contributor: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    depth_margin: float,
) -> float:
    expected = _float(record.get("expected_depth"), 0.0)
    rendered = _float(record.get("rendered_depth"), 0.0)
    if expected <= 0.0 or rendered <= 0.0:
        return 1.0
    if rendered >= expected - float(depth_margin):
        return 0.0
    if "depth" not in contributor and "z" not in contributor:
        return 1.0
    depth = _float(contributor.get("depth", contributor.get("z")), rendered)
    return 1.0 if depth <= expected - float(depth_margin) else 0.0


def _metadata_value(record: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def accumulate_rendered_feedback(
    observations: Iterable[Mapping[str, Any]],
    *,
    num_gaussians: int,
    split_name: str,
    mode: str = "observed_only",
    reprojection_threshold_px: float = 8.0,
    weak_outlier_weight: float = 0.02,
    artifact_weight: float = 1.0,
    depth_margin: float = 1.0,
) -> dict[str, Any]:
    """Build export-compatible Gaussian support/risk tensors from source-ray observations.

    The function is intentionally renderer-free. In ``observed_only`` mode it accepts the
    same contributor records as existing observation JSONL files. In
    ``rendered_depth_augmented`` mode, optional rendered/expected depth and artifact scores
    become weighted per-Gaussian artifact-risk contributions.
    """

    split = validate_split_name(split_name)
    normalized_mode = validate_mode(mode)
    count = int(num_gaussians)
    if count < 0:
        raise ValueError(f"num_gaussians must be non-negative, got {count}")

    observed_count = torch.zeros(count, dtype=torch.long)
    positive_observed_count = torch.zeros(count, dtype=torch.long)
    contribution_mass = torch.zeros(count, dtype=torch.float32)
    positive_contribution_mass = torch.zeros(count, dtype=torch.float32)
    weighted_support = torch.zeros(count, dtype=torch.float32)
    hard_negative_contribution = torch.zeros(count, dtype=torch.float32)
    artifact_risk_contribution = torch.zeros(count, dtype=torch.float32)
    source_views: set[str] = set()
    query_ids: set[str] = set()
    split_names_seen: set[str] = set()
    observation_count = 0
    rendered_observation_count = 0
    skipped_empty_contributor_count = 0

    for record in observations:
        record_split = _record_split(record, split)
        split_names_seen.add(record_split)
        contributors = _contributors(record)
        normalized = normalize_contributors(contributors)
        if not normalized:
            skipped_empty_contributor_count += 1
            continue

        observation_count += 1
        source_id = _metadata_value(record, "source_view_id", "view_id")
        query_id = _metadata_value(record, "query_id", "image_id")
        if source_id:
            source_views.add(source_id)
        if query_id:
            query_ids.add(query_id)

        pnp_inlier = _bool(record.get("pnp_inlier", False))
        quality = _observation_quality(record, float(reprojection_threshold_px)) if pnp_inlier else 0.0
        reproj = _float(record.get("reprojection_error_px"), 0.0)
        descriptor = _float(record.get("descriptor_score"), 0.0)
        weak_risk = (
            float(weak_outlier_weight)
            if (not pnp_inlier and descriptor >= 0.5 and reproj > float(reprojection_threshold_px))
            else 0.0
        )

        artifact = 0.0
        raw_by_gid: dict[int, Mapping[str, Any]] = {}
        if normalized_mode == "rendered_depth_augmented":
            if "rendered_depth" in record or "expected_depth" in record or "artifact_score" in record:
                rendered_observation_count += 1
            artifact = _artifact_score(record, depth_margin=float(depth_margin))
            raw_by_gid = {
                int(item["gaussian_id"]): raw
                for item in normalized
                for raw in contributors
                if int(item["gaussian_id"]) == int(raw.get("gaussian_id", raw.get("matched_gaussian_id", -1)))
            }

        for item in normalized:
            gid = int(item["gaussian_id"])
            if gid >= count:
                raise ValueError(f"gaussian_id {gid} is outside num_gaussians={count}")
            weight = float(item["weight"])
            observed_count[gid] += 1
            contribution_mass[gid] += weight
            hard_negative_contribution[gid] += weak_risk * weight
            if pnp_inlier and quality > 0.0:
                positive_observed_count[gid] += 1
                positive_contribution_mass[gid] += weight
                weighted_support[gid] += quality * weight
            if artifact > 0.0:
                raw = raw_by_gid.get(gid, {})
                factor = _artifact_factor(raw, record, depth_margin=float(depth_margin))
                artifact_risk_contribution[gid] += float(artifact_weight) * artifact * weight * factor

    hard_negative_risk = hard_negative_contribution / contribution_mass.clamp_min(1e-6)
    artifact_risk = artifact_risk_contribution / contribution_mass.clamp_min(1e-6)
    support_score = weighted_support
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": normalized_mode,
        "split_name": split,
        "split_names_seen": sorted(split_names_seen),
        "num_gaussians": int(count),
        "observation_count": int(observation_count),
        "rendered_observation_count": int(rendered_observation_count),
        "skipped_empty_contributor_count": int(skipped_empty_contributor_count),
        "query_count": int(len(query_ids)),
        "source_view_count": int(len(source_views)),
        "hyperparameters": {
            "reprojection_threshold_px": float(reprojection_threshold_px),
            "weak_outlier_weight": float(weak_outlier_weight),
            "artifact_weight": float(artifact_weight),
            "depth_margin": float(depth_margin),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "support_score": support_score.to(torch.float32),
        "weighted_support": weighted_support.to(torch.float32),
        "contribution_mass": contribution_mass.to(torch.float32),
        "positive_contribution_mass": positive_contribution_mass.to(torch.float32),
        "observed_count": observed_count,
        "positive_observed_count": positive_observed_count,
        "hard_negative_risk": hard_negative_risk.to(torch.float32).clamp(0.0, 1.0),
        "hard_negative_contribution": hard_negative_contribution.to(torch.float32),
        "artifact_risk": artifact_risk.to(torch.float32).clamp(0.0, 1.0),
        "artifact_risk_contribution": artifact_risk_contribution.to(torch.float32),
        "metadata": metadata,
    }
