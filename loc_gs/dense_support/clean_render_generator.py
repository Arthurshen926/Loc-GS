from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CleanRenderPolicy:
    """Scoring policy for sparse-anchored clean dense render selection.

    This policy selects a render candidate only. It does not accept, reject, or
    interpolate a final localization pose.
    """

    min_visible_ratio: float = 0.35
    min_coverage_grid_cells: int = 4
    min_feature_cosine: float = 0.08
    min_score_gain: float = 0.05
    translation_penalty_per_m: float = 0.04
    gating_removed_penalty: float = 0.75
    failed_check_penalty: float = 0.35
    max_anchor_loss_ratio: float = 0.50


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _translation_norm_m(candidate: Mapping[str, Any]) -> float:
    fast_score = candidate.get("fast_score")
    if isinstance(fast_score, Mapping):
        value = _as_float(fast_score.get("translation_norm_m"), 0.0)
        if value > 0.0:
            return value
    translation = candidate.get("translation_cam_m")
    if translation is None:
        return 0.0
    try:
        return float(np.linalg.norm(np.asarray(translation, dtype=np.float64).reshape(-1)))
    except (TypeError, ValueError):
        return 0.0


def _gating_removed_fraction(gating: Mapping[str, Any] | None) -> float:
    if not isinstance(gating, Mapping):
        return 0.0
    conflict_count = max(_as_float(gating.get("conflict_count"), 0.0), 0.0)
    kept_count = max(_as_float(gating.get("kept_count"), 0.0), 0.0)
    total = conflict_count + kept_count
    if total <= 1.0e-12:
        return 0.0
    return float(np.clip(conflict_count / total, 0.0, 1.0))


def _candidate_gating(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    render_control = candidate.get("render_control")
    if isinstance(render_control, Mapping):
        gating = render_control.get("gaussian_gating")
        if isinstance(gating, Mapping):
            return gating
    gating = candidate.get("gaussian_gating")
    return gating if isinstance(gating, Mapping) else None


def _is_base_label(label: Any) -> bool:
    return str(label or "").strip() in {"base", "native", "native_base", "base_native"}


def _failed_check_count(preflight: Mapping[str, Any]) -> int:
    failed = preflight.get("failed_checks")
    if not isinstance(failed, Mapping):
        return 0
    return int(sum(1 for value in failed.values() if bool(value)))


def _soft_ratio(value: Any, target: float) -> float:
    observed = _as_float(value, 0.0)
    if float(target) <= 1.0e-12:
        return 1.0
    return float(np.clip(observed / float(target), 0.0, 1.0))


def _anchor_safe(preflight: Mapping[str, Any], policy: CleanRenderPolicy) -> bool:
    if preflight.get("sparse_confident") is False:
        return False
    visible_ratio = _as_float(preflight.get("visible_ratio"), 0.0)
    coverage = _as_int(preflight.get("coverage_grid_cells"), 0)
    decision = str(preflight.get("decision", ""))
    if visible_ratio < float(policy.min_visible_ratio) * (1.0 - float(policy.max_anchor_loss_ratio)):
        return False
    if coverage < max(1, int(policy.min_coverage_grid_cells) // 2):
        return False
    return decision != "retry_sparse_or_patch_dense"


def score_clean_render_candidate(
    candidate: Mapping[str, Any],
    *,
    policy: CleanRenderPolicy = CleanRenderPolicy(),
) -> dict[str, Any]:
    """Score one dense render candidate without selecting a final pose."""

    preflight = candidate.get("preflight")
    if not isinstance(preflight, Mapping):
        preflight = {}
    translation_norm = _translation_norm_m(candidate)
    gating_fraction = _gating_removed_fraction(_candidate_gating(candidate))
    visible_ratio = _as_float(preflight.get("visible_ratio"), 0.0)
    coverage = _as_int(preflight.get("coverage_grid_cells"), 0)
    feature = max(_as_float(preflight.get("feature_cosine_median"), 0.0), 0.0)
    local_variance = max(_as_float(preflight.get("local_feature_variance_median"), 0.0), 0.0)
    near_occluder_ratio = float(np.clip(_as_float(preflight.get("near_occluder_ratio"), 0.0), 0.0, 1.0))
    sparse_inliers = _as_int(preflight.get("sparse_inlier_count"), 0)
    anchor_safe = _anchor_safe(preflight, policy)
    failed_count = _failed_check_count(preflight)

    anchor_score = 2.0 * _soft_ratio(visible_ratio, float(policy.min_visible_ratio))
    anchor_score += 0.12 * min(float(coverage), 16.0)
    query_render_score = 1.5 * _soft_ratio(feature, float(policy.min_feature_cosine))
    query_render_score += 0.25 * min(local_variance / 0.02, 1.0)
    artifact_score = 1.0 - near_occluder_ratio
    score = anchor_score + query_render_score + artifact_score
    score -= float(policy.failed_check_penalty) * failed_count
    score -= float(policy.translation_penalty_per_m) * translation_norm
    score -= float(policy.gating_removed_penalty) * gating_fraction
    if not anchor_safe:
        score -= 3.0
    if str(preflight.get("decision")) == "accept_dense":
        score += 0.25
    return {
        "label": str(candidate.get("label", "candidate")),
        "score": float(score),
        "anchor_score": float(anchor_score),
        "query_render_score": float(query_render_score),
        "artifact_score": float(artifact_score),
        "translation_norm_m": float(translation_norm),
        "gating_removed_fraction": float(gating_fraction),
        "anchor_safe": bool(anchor_safe),
        "failed_check_count": int(failed_count),
        "visible_ratio": float(visible_ratio),
        "coverage_grid_cells": int(coverage),
        "feature_cosine_median": float(feature),
        "near_occluder_ratio": float(near_occluder_ratio),
        "sparse_inlier_count": int(sparse_inliers),
        "preflight_decision": str(preflight.get("decision", "unknown")),
    }


def select_clean_render_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: CleanRenderPolicy = CleanRenderPolicy(),
) -> dict[str, Any]:
    """Select a cleaner render candidate for dense matching/refinement.

    The selected pose is a render pose. The result intentionally marks that it
    does not select a final localization pose, keeping clean render generation
    separate from dense correspondence generation and pose refinement.
    """

    if not candidates:
        return {
            "schema": "loc_gs_clean_render_generator_v1",
            "decision": "no_render_candidate",
            "candidate_count": 0,
            "selected_label": None,
            "selected_role": "render_pose_only",
            "does_not_select_final_pose": True,
            "diagnostic_only": True,
        }
    candidate_scores = [score_clean_render_candidate(candidate, policy=policy) for candidate in candidates]
    base_idx = next(
        (idx for idx, score in enumerate(candidate_scores) if _is_base_label(score.get("label"))),
        0,
    )
    base_missing = not _is_base_label(candidate_scores[base_idx].get("label"))
    base_score = candidate_scores[base_idx]
    safe_indices = [idx for idx, score in enumerate(candidate_scores) if bool(score["anchor_safe"])]
    if safe_indices:
        best_idx = max(safe_indices, key=lambda idx: candidate_scores[idx]["score"])
    else:
        best_idx = base_idx
    best_score = candidate_scores[best_idx]
    score_gain = float(best_score["score"] - base_score["score"])
    if best_idx == base_idx or score_gain < float(policy.min_score_gain):
        selected_idx = base_idx
        decision = "use_native_clean_render"
    else:
        selected_idx = best_idx
        decision = "use_clean_render_candidate"
    selected = dict(candidates[selected_idx])
    selected_score = candidate_scores[selected_idx]
    selected_pose = selected.get("pose_w2c")
    selected_pose_json = None
    if selected_pose is not None:
        try:
            selected_pose_json = np.asarray(selected_pose, dtype=np.float64).reshape(4, 4).tolist()
        except (TypeError, ValueError):
            selected_pose_json = None
    return {
        "schema": "loc_gs_clean_render_generator_v1",
        "decision": decision,
        "selected_label": str(selected_score["label"]),
        "selected_index": int(selected_idx),
        "base_index": int(base_idx),
        "base_missing": bool(base_missing),
        "selected_score": float(selected_score["score"]),
        "base_score": float(base_score["score"]),
        "score_gain": float(selected_score["score"] - base_score["score"]),
        "candidate_count": int(len(candidates)),
        "candidate_scores": candidate_scores,
        "selected": selected_score,
        "selected_pose_w2c": selected_pose_json,
        "selected_gating": _candidate_gating(selected),
        "selected_role": "render_pose_only",
        "does_not_select_final_pose": True,
        "policy": {
            "min_visible_ratio": float(policy.min_visible_ratio),
            "min_coverage_grid_cells": int(policy.min_coverage_grid_cells),
            "min_feature_cosine": float(policy.min_feature_cosine),
            "min_score_gain": float(policy.min_score_gain),
            "translation_penalty_per_m": float(policy.translation_penalty_per_m),
            "gating_removed_penalty": float(policy.gating_removed_penalty),
            "failed_check_penalty": float(policy.failed_check_penalty),
            "max_anchor_loss_ratio": float(policy.max_anchor_loss_ratio),
        },
        "diagnostic_only": True,
    }
