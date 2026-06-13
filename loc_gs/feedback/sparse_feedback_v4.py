from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from loc_gs.feedback.solver_causal_attribution import (
    annotate_sparse_correspondences_with_solver_causal_fields,
    pnp_conditioning_metrics,
)


SCHEMA_VERSION = "sparse_solver_feedback_v4"
LABEL_ROLES = (
    "protected_support",
    "positive_inlier",
    "harmful_negative",
    "risky_competitor_negative",
    "neutral_outlier",
)
VALID_SOURCE_ROLES = {"baseline_trace", "candidate_trace", "render_aug_trace"}
REQUIRED_QUERY_FIELDS = (
    "scene",
    "split_name",
    "query_id",
    "image_id",
    "pose_success",
    "sparse_te_cm",
    "sparse_re_deg",
    "catastrophic_failure",
    "inlier_count",
    "match_count",
    "inlier_image_cell_count",
    "inlier_depth_bin_count",
    "bearing_spread",
    "depth_spread",
)
REQUIRED_CORRESPONDENCE_FIELDS = (
    "scene",
    "split_name",
    "query_id",
    "image_id",
    "gaussian_id",
    "sampled_row",
    "query_keypoint_index",
    "keypoint_xy",
    "query_xy_norm",
    "image_cell",
    "descriptor_score",
    "descriptor_margin",
    "detector_score",
    "pnp_inlier",
    "reprojection_error_px",
    "camera_xyz",
    "depth_m",
    "bearing",
    "source_role",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if math.isfinite(number) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        raise ValueError("split_name is required")
    split_audit = payload.get("split_audit", {})
    official_test_used = False
    test_split_used = False
    audit_split = ""
    if isinstance(split_audit, Mapping):
        official_test_used = bool(split_audit.get("official_test_used", False))
        test_split_used = bool(split_audit.get("test_split_used", False))
        audit_split = str(split_audit.get("split_name", "")).strip()
    if bool(payload.get("official_test_used", False)) or bool(payload.get("test_split_used", False)):
        official_test_used = True
    if split.lower() == "test" or audit_split.lower() == "test" or test_split_used or official_test_used:
        raise ValueError("test split is not allowed for sparse solver feedback v4")
    return split


def _missing_required(row: Mapping[str, Any], required: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for key in required:
        if key not in row or row.get(key) is None:
            missing.append(key)
    return missing


def _validate_query_row(row: Mapping[str, Any]) -> None:
    missing = _missing_required(row, REQUIRED_QUERY_FIELDS)
    if missing:
        raise ValueError(f"query row is missing required field {missing[0]}")


def _validate_correspondence_row(row: Mapping[str, Any]) -> None:
    missing = _missing_required(row, REQUIRED_CORRESPONDENCE_FIELDS)
    if missing:
        raise ValueError(f"correspondence row is missing required field {missing[0]}")
    source_role = str(row.get("source_role", "")).strip()
    if source_role not in VALID_SOURCE_ROLES:
        raise ValueError(f"correspondence source_role must be one of {sorted(VALID_SOURCE_ROLES)}")


def _validate_row_split(row: Mapping[str, Any], *, expected_split: str, row_type: str) -> None:
    row_split = str(row.get("split_name", "")).strip()
    if row_split.lower() == "test":
        raise ValueError(f"test split is not allowed in {row_type} row")
    if row_split and row_split != str(expected_split):
        raise ValueError(f"{row_type} split_name mismatch: expected {expected_split}, got {row_split}")


def _query_state(row: Mapping[str, Any], *, protected_te_cm: float, hard_te_cm: float) -> dict[str, Any]:
    sparse_te_cm = _finite_float(row.get("sparse_te_cm"), 1e9)
    raw_pose_success = _bool(row.get("pose_success"))
    pose_success = bool(raw_pose_success and sparse_te_cm <= float(hard_te_cm))
    protected_good_query = bool(pose_success and sparse_te_cm <= float(protected_te_cm))
    hard_query = bool((not pose_success) or sparse_te_cm > float(hard_te_cm))
    inliers = max(0.0, _finite_float(row.get("inlier_count"), 0.0))
    matches = max(1.0, _finite_float(row.get("match_count"), 1.0))
    out = dict(row)
    out.update(
        {
            "pose_success": pose_success,
            "protected_good_query": protected_good_query,
            "hard_query": hard_query,
            "inlier_ratio": float(inliers / matches),
        }
    )
    return out


def _label_correspondence(
    row: Mapping[str, Any],
    query: Mapping[str, Any],
    *,
    low_reprojection_px: float,
    high_score_threshold: float,
    harmful_regression_cm: float,
    min_descriptor_margin: float,
    risky_competitor_max_margin: float,
) -> str:
    pnp_inlier = _bool(row.get("pnp_inlier"))
    reprojection_error_px = _finite_float(row.get("reprojection_error_px"), 1e9)
    descriptor_score = _finite_float(row.get("descriptor_score"), 0.0)
    descriptor_margin = _finite_float(row.get("descriptor_margin"), 0.0)
    regression_cm = _finite_float(query.get("candidate_minus_baseline_te_cm"), 0.0)
    low_reproj = reprojection_error_px <= float(low_reprojection_px)
    sufficient_margin = descriptor_margin >= float(min_descriptor_margin)
    if pnp_inlier and bool(query.get("protected_good_query", False)) and low_reproj and sufficient_margin:
        return "protected_support"
    if pnp_inlier and bool(query.get("pose_success", False)) and low_reproj and sufficient_margin:
        return "positive_inlier"
    if (not pnp_inlier) and bool(query.get("hard_query", False)) and descriptor_score >= float(high_score_threshold):
        return "harmful_negative"
    if (not pnp_inlier) and regression_cm >= float(harmful_regression_cm) and descriptor_score >= float(high_score_threshold):
        return "harmful_negative"
    if (
        (not pnp_inlier)
        and bool(query.get("protected_good_query", False))
        and descriptor_score >= float(high_score_threshold)
        and descriptor_margin <= float(risky_competitor_max_margin)
    ):
        return "risky_competitor_negative"
    return "neutral_outlier"


def _negative_reason(label_role: str) -> str:
    if label_role == "harmful_negative":
        return "hard_or_regression_high_score_outlier"
    if label_role == "risky_competitor_negative":
        return "protected_good_high_score_low_margin_outlier"
    return ""


def build_sparse_feedback_v4(
    payload: Mapping[str, Any],
    *,
    protected_te_cm: float = 10.0,
    hard_te_cm: float = 20.0,
    low_reprojection_px: float = 4.0,
    high_score_threshold: float = 0.8,
    harmful_regression_cm: float = 20.0,
    min_descriptor_margin: float = 0.05,
    risky_competitor_max_margin: float = 0.05,
) -> dict[str, Any]:
    split = _split_name(payload)
    query_rows = [dict(row) for row in payload.get("queries", []) if isinstance(row, Mapping)]
    for row in query_rows:
        _validate_query_row(row)
        _validate_row_split(row, expected_split=split, row_type="query")
    query_index = {
        str(row.get("query_id")): _query_state(
            row,
            protected_te_cm=float(protected_te_cm),
            hard_te_cm=float(hard_te_cm),
        )
        for row in query_rows
        if str(row.get("query_id", "")).strip()
    }

    label_counts = {label: 0 for label in LABEL_ROLES}
    correspondences: list[dict[str, Any]] = []
    raw_correspondences = [raw for raw in payload.get("correspondences", []) if isinstance(raw, Mapping)]
    for raw in raw_correspondences:
        _validate_correspondence_row(raw)
        _validate_row_split(raw, expected_split=split, row_type="correspondence")
    causal_annotations, causal_metrics = annotate_sparse_correspondences_with_solver_causal_fields(raw_correspondences)
    query_conditioning_rows: dict[str, list[Mapping[str, Any]]] = {}
    for raw in raw_correspondences:
        if _bool(raw.get("pnp_inlier", False)):
            query_conditioning_rows.setdefault(str(raw.get("query_id", "")), []).append(raw)
    for query_id, rows in query_conditioning_rows.items():
        if query_id in query_index:
            query_index[query_id].update(pnp_conditioning_metrics(rows))

    for raw, causal_annotation in zip(raw_correspondences, causal_annotations):
        if not isinstance(raw, Mapping):
            continue
        query_id = str(raw.get("query_id", ""))
        query = query_index.get(query_id, {})
        label = _label_correspondence(
            raw,
            query,
            low_reprojection_px=float(low_reprojection_px),
            high_score_threshold=float(high_score_threshold),
            harmful_regression_cm=float(harmful_regression_cm),
            min_descriptor_margin=float(min_descriptor_margin),
            risky_competitor_max_margin=float(risky_competitor_max_margin),
        )
        row = dict(raw)
        row["label_role"] = label
        row["negative_reason"] = _negative_reason(label)
        row["query_protected_good"] = bool(query.get("protected_good_query", False))
        row["query_hard"] = bool(query.get("hard_query", False))
        row["query_pose_success"] = bool(query.get("pose_success", False))
        row["query_sparse_te_cm"] = _finite_float(query.get("sparse_te_cm"), 1e9)
        row.update(causal_annotation)
        correspondences.append(row)
        label_counts[label] += 1

    split_audit = {
        "schema_version": "sparse_solver_feedback_v4_split_audit_v1",
        "split_name": split,
        "test_split_used": False,
        "official_test_used": False,
        "role": "sparse_solver_feedback_v4",
    }
    query_features = payload.get("query_features", {})
    if not isinstance(query_features, Mapping):
        query_features = {}
    query_match_descriptors = payload.get("query_match_descriptors", {})
    query_match_descriptor_query_count = len(query_match_descriptors) if isinstance(query_match_descriptors, Mapping) else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "query_index": query_index,
        "query_features": dict(query_features),
        "query_match_descriptor_query_count": int(query_match_descriptor_query_count),
        "correspondences": correspondences,
        "metrics": {
            "query_count": int(len(query_index)),
            "correspondence_count": int(len(correspondences)),
            **{f"{label}_count": int(count) for label, count in label_counts.items()},
            **causal_metrics,
        },
        "split_audit": split_audit,
    }
