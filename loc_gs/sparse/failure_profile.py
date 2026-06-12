from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class SparseFailureProfileConfig:
    selected_correct_ratio_floor: float = 0.10
    inlier_correct_ratio_floor: float = 0.50
    target_median_te_cm: float = 10.0


def build_sparse_failure_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    scene: str,
    split_name: str,
    cfg: SparseFailureProfileConfig | None = None,
) -> dict[str, object]:
    if cfg is None:
        cfg = SparseFailureProfileConfig()
    split = reject_test_split(split_name, purpose="internal sparse failure profile")
    mode_counts: dict[str, int] = {}
    per_query: list[dict[str, object]] = []
    post_changed_sum = 0
    post_corrected_sum = 0
    post_worsened_sum = 0
    post_delta_sum = 0
    for row in rows:
        modes = _query_failure_modes(row, cfg)
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        post_changed_sum += _as_int(row.get("post_pnp_rescore_changed_count"))
        post_corrected_sum += _as_int(row.get("post_pnp_rescore_corrected_count"))
        post_worsened_sum += _as_int(row.get("post_pnp_rescore_worsened_count"))
        post_delta_sum += _as_int(row.get("post_pnp_rescore_correct_delta"))
        selected = dict(row.get("selected_set_diagnostics", {}) or {})
        inlier = dict(row.get("inlier_set_diagnostics", {}) or {})
        per_query.append(
            {
                "query_id": str(row.get("query_id", "")),
                "success": bool(row.get("success", False)),
                "te_cm": _maybe_float(row.get("te_cm")),
                "failure_modes": modes,
                "selected_geometric_correct_ratio": _maybe_float(
                    selected.get("selected_geometric_correct_ratio")
                ),
                "inlier_geometric_correct_ratio": _maybe_float(inlier.get("inlier_geometric_correct_ratio")),
                "post_pnp_rescore_correct_delta": _as_int(row.get("post_pnp_rescore_correct_delta")),
                "set_conflict_rerank_changed_count": _as_int(row.get("set_conflict_rerank_changed_count")),
            }
        )
    median_te = _maybe_float(metrics.get("median_te_cm"))
    target = float(cfg.target_median_te_cm)
    profile = {
        "schema_version": "internal_sparse_failure_profile_v1",
        "scene": str(scene),
        "split_name": split,
        "query_count": int(len(rows)),
        "success_count": _as_int(metrics.get("success_count")),
        "median_te_cm": median_te,
        "median_re_deg": _maybe_float(metrics.get("median_re_deg")),
        "target_median_te_cm": target,
        "target_gap_cm": None if median_te is None else float(median_te - target),
        "failure_mode_counts": {key: int(mode_counts[key]) for key in sorted(mode_counts)},
        "dominant_failure_modes": [
            {"mode": key, "count": int(count)}
            for key, count in sorted(mode_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ],
        "candidate_artifact": _compact_candidate_artifact(metrics.get("candidate_artifact")),
        "rerank_diagnostic": _compact_rerank(metrics),
        "post_pnp_rescore": {
            "enabled": bool(metrics.get("post_pnp_candidate_rescore_enabled", False)),
            "changed_sum": int(post_changed_sum),
            "corrected_sum": int(post_corrected_sum),
            "worsened_sum": int(post_worsened_sum),
            "correct_delta_sum": int(post_delta_sum),
            "max_score_drop": metrics.get("post_pnp_rescore_max_score_drop"),
        },
        "recommendation": _recommendation(mode_counts),
        "thresholds": asdict(cfg),
        "per_query": per_query,
    }
    return profile


def _query_failure_modes(row: Mapping[str, Any], cfg: SparseFailureProfileConfig) -> list[str]:
    if not bool(row.get("success", False)):
        return ["missing_or_failed_pose"]
    modes: list[str] = []
    selected = dict(row.get("selected_set_diagnostics", {}) or {})
    inlier = dict(row.get("inlier_set_diagnostics", {}) or {})
    selected_ratio = _maybe_float(selected.get("selected_geometric_correct_ratio"))
    inlier_ratio = _maybe_float(inlier.get("inlier_geometric_correct_ratio"))
    if selected_ratio is not None and selected_ratio < float(cfg.selected_correct_ratio_floor):
        modes.append("selected_set_low_precision")
    if inlier_ratio is not None and inlier_ratio < float(cfg.inlier_correct_ratio_floor):
        modes.append("inlier_set_wrong_dominant")
    corrected = _as_int(row.get("post_pnp_rescore_corrected_count"))
    worsened = _as_int(row.get("post_pnp_rescore_worsened_count"))
    delta = _as_int(row.get("post_pnp_rescore_correct_delta"))
    changed = _as_int(row.get("post_pnp_rescore_changed_count"))
    if changed > 0 and (delta < 0 or worsened > corrected):
        modes.append("post_pnp_rescore_harm")
    return modes


def _compact_candidate_artifact(value: Any) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    keys = ("top1_correct", "topk_available", "oracle_gap", "positive_ratio", "topk", "keypoint_count")
    return {key: value[key] for key in keys if key in value}


def _compact_rerank(metrics: Mapping[str, Any]) -> dict[str, object]:
    keys = (
        "rerank_diagnostic_enabled",
        "native_top1_correct",
        "reranked_top1_correct",
        "reranked_top1_gain",
        "reranked_top1_changed_count",
        "reranked_topk_available",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _recommendation(mode_counts: Mapping[str, int]) -> str:
    if mode_counts.get("selected_set_low_precision", 0) or mode_counts.get("inlier_set_wrong_dominant", 0):
        return "prioritize_set_level_selection_and_inlier_precision"
    if mode_counts.get("post_pnp_rescore_harm", 0):
        return "disable_or_constrain_post_pnp_rescore"
    if mode_counts.get("missing_or_failed_pose", 0):
        return "audit_candidate_coverage_and_camera_inputs"
    return "monitor_no_dominant_failure"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
