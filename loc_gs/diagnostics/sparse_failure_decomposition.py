from __future__ import annotations

import math
from typing import Any


CAUSES = (
    "detector_starvation",
    "descriptor_ambiguity",
    "ranking_failure",
    "geometry_degeneracy",
    "map_or_render_artifact",
)


def _float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _low_score(value: float, good: float, bad: float) -> float:
    if value >= good:
        return 0.0
    if value <= bad:
        return 1.0
    return _clip01((good - value) / (good - bad))


def decompose_sparse_failure(payload: dict[str, Any]) -> dict[str, Any]:
    """Attribute a sparse localization failure to observable mechanisms.

    This is a diagnostic classifier, not a training label generator. It uses
    only observable match/PnP/render summary statistics and can be run on
    train/dev or test for reporting, but test outputs must not feed method
    selection.
    """

    keypoints = _float(payload, "keypoint_count", default=250.0)
    matches = _float(payload, "match_count", default=80.0)
    inliers = _float(payload, "pnp_inlier_count", default=20.0)
    coverage = _float(payload, "image_coverage", default=1.0)
    descriptor_margin = _float(payload, "descriptor_margin_median", default=1.0)
    conflict_rate = _float(payload, "descriptor_conflict_rate")
    top_rank_inlier_rate = _float(payload, "top_rank_inlier_rate", default=1.0)
    all_match_inlier_rate = _float(payload, "all_match_inlier_rate", default=inliers / max(matches, 1.0))
    oracle_inliers = _float(payload, "oracle_inlier_count", default=inliers)
    logdet = _float(payload, "geometry_logdet", default=10.0)
    min_eig = _float(payload, "geometry_min_eigenvalue", default=1.0)
    depth_spread = _float(payload, "depth_spread_m", default=1.0)
    render_valid = _float(payload, "render_valid_ratio", default=1.0)
    render_cosine = _float(payload, "render_feature_cosine_median", default=1.0)
    artifact_risk = _float(payload, "artifact_risk")

    detector_score = max(
        _low_score(keypoints, good=250.0, bad=80.0),
        _low_score(matches, good=80.0, bad=20.0),
        _low_score(inliers, good=20.0, bad=6.0),
        _low_score(coverage, good=0.25, bad=0.08),
    )
    descriptor_score = max(
        _low_score(descriptor_margin, good=0.15, bad=0.03),
        _clip01((conflict_rate - 0.25) / 0.35),
    )
    ranking_gap = 0.0
    if oracle_inliers > max(inliers * 1.4, inliers + 12.0):
        ranking_gap = 0.6
    if all_match_inlier_rate > 0.0 and top_rank_inlier_rate < all_match_inlier_rate * 0.5:
        ranking_gap = max(ranking_gap, 0.9)
    ranking_score = ranking_gap
    geometry_score = max(
        _low_score(logdet, good=4.0, bad=0.5),
        _low_score(min_eig, good=0.05, bad=0.005),
        _low_score(depth_spread, good=0.8, bad=0.1),
    )
    artifact_score = max(
        _low_score(render_valid, good=0.7, bad=0.3),
        _low_score(render_cosine, good=0.2, bad=0.05),
        artifact_risk,
    )

    scores = {
        "detector_starvation": round(detector_score, 6),
        "descriptor_ambiguity": round(descriptor_score, 6),
        "ranking_failure": round(ranking_score, 6),
        "geometry_degeneracy": round(geometry_score, 6),
        "map_or_render_artifact": round(artifact_score, 6),
    }
    active_flags = [cause for cause in CAUSES if scores[cause] >= 0.5]
    if not active_flags:
        active_flags = ["unresolved_or_mild"]

    priority = {
        "map_or_render_artifact": 4,
        "detector_starvation": 3,
        "descriptor_ambiguity": 2,
        "ranking_failure": 1,
        "geometry_degeneracy": 0,
    }
    primary = max(CAUSES, key=lambda cause: (scores[cause], priority[cause]))
    if scores[primary] < 0.5:
        primary = "unresolved_or_mild"

    return {
        "schema_version": "sparse_failure_decomposition_v1",
        "query_id": str(payload.get("query_id", "")),
        "primary_cause": primary,
        "active_flags": active_flags,
        "scores": scores,
        "observations": {
            "keypoint_count": keypoints,
            "match_count": matches,
            "pnp_inlier_count": inliers,
            "oracle_inlier_count": oracle_inliers,
            "all_match_inlier_rate": all_match_inlier_rate,
            "top_rank_inlier_rate": top_rank_inlier_rate,
            "image_coverage": coverage,
            "descriptor_margin_median": descriptor_margin,
            "descriptor_conflict_rate": conflict_rate,
            "geometry_logdet": logdet,
            "depth_spread_m": depth_spread,
            "render_valid_ratio": render_valid,
            "render_feature_cosine_median": render_cosine,
        },
    }
