from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping


POSITIVE_LABEL_ROLES = {"protected_support", "positive_inlier"}
NEGATIVE_LABEL_ROLES = {"harmful_negative", "risky_competitor_negative"}
NEUTRAL_LABEL_ROLES = {"neutral_outlier"}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _label(row: Mapping[str, Any]) -> int | None:
    raw = row.get("label", row.get("impact_label"))
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _label_role(row: Mapping[str, Any]) -> str:
    return str(row.get("label_role", row.get("role", ""))).strip()


def _is_test_split_name(split: str) -> bool:
    lowered = str(split).strip().lower()
    return lowered == "test" or lowered == "official_test" or lowered.endswith("_test")


def _id_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value)
        digits = []
        for char in reversed(text):
            if char.isdigit() or (char == "-" and digits):
                digits.append(char)
            elif digits:
                break
        if not digits:
            return int(default)
        try:
            return int("".join(reversed(digits)))
        except ValueError:
            return int(default)


def _audit_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = payload.get("split_audit", {})
    return audit if isinstance(audit, Mapping) else {}


def _split_name(payload: Mapping[str, Any], records: list[Mapping[str, Any]]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    manifest = payload.get("manifest")
    if not split and isinstance(manifest, Mapping):
        split = str(manifest.get("split_name", manifest.get("split", ""))).strip()
    audit = _audit_mapping(payload)
    if not split:
        split = str(audit.get("split_name", audit.get("split", ""))).strip()
    if not split:
        for row in records:
            split = str(row.get("split_name", row.get("split", ""))).strip()
            if split:
                break
    return split


def _validate_split_audit(payload: Mapping[str, Any], split: str) -> None:
    audit = _audit_mapping(payload)
    if bool(audit.get("test_split_used", False)) or bool(audit.get("official_test_used", False)):
        raise ValueError("refusing to build solver feedback impact from test split")
    audit_split = str(audit.get("split_name", audit.get("split", ""))).strip()
    if _is_test_split_name(audit_split):
        raise ValueError("refusing to build solver feedback impact from test split")
    if audit_split and split and audit_split != split:
        raise ValueError(f"split_name mismatch: expected {split}, got audit split {audit_split}")


def _positive_weight(row: Mapping[str, Any]) -> float:
    role = _label_role(row)
    if role:
        if role not in POSITIVE_LABEL_ROLES:
            return 0.0
    else:
        label = _label(row)
        if label is not None and label != 1:
            return 0.0
    label = _label(row)
    if not _bool_value(row.get("pnp_inlier"), default=False):
        return 0.0
    reproj = max(0.0, _f(row.get("reprojection_error_px"), 999.0))
    margin = max(0.0, _f(row.get("descriptor_margin"), 0.0))
    pose_ok = _bool_value(row.get("pose_success", row.get("pnp_success")), default=False)
    pose = 1.0 if pose_ok else 0.5
    return pose * (1.0 / (1.0 + reproj / 4.0)) * (0.5 + min(0.5, margin))


def _negative_weight(row: Mapping[str, Any]) -> float:
    role = _label_role(row)
    if role:
        if role not in NEGATIVE_LABEL_ROLES:
            return 0.0
    else:
        label = _label(row)
        if label is not None and label != 0:
            return 0.0
    if _bool_value(row.get("pnp_inlier"), default=False):
        return 0.0
    reproj = max(0.0, _f(row.get("reprojection_error_px"), 0.0))
    score = max(0.0, _f(row.get("descriptor_score"), 0.0))
    regression = max(
        0.0,
        _f(
            row.get(
                "query_regression_delta_cm",
                row.get("query_sparse_te_cm", row.get("pose_error_t_cm")),
            ),
            0.0,
        ),
    )
    margin = max(0.0, _f(row.get("descriptor_margin"), 0.0))
    low_margin = 1.0 / (1.0 + margin / 0.15)
    return (0.25 + score) * min(2.0, reproj / 8.0) * (1.0 + min(2.0, regression / 20.0)) * low_margin


def _detector_entry(row: Mapping[str, Any], gid: int, weight: float) -> dict[str, Any]:
    return {
        "gaussian_id": int(gid),
        "image_id": str(row.get("image_id", row.get("source_view_id", ""))),
        "keypoint_xy": row.get("keypoint_xy"),
        "query_xy_norm": row.get("query_xy_norm"),
        "source_role": str(row.get("source_role", "")),
        "weight": float(weight),
    }


def build_solver_feedback_impact(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_records = payload.get("records", payload.get("correspondences", []))
    records = [row for row in raw_records if isinstance(row, Mapping)]
    split = _split_name(payload, records)
    if not split:
        raise ValueError("split_name is required")
    if _is_test_split_name(split):
        raise ValueError("refusing to build solver feedback impact from test split")
    _validate_split_audit(payload, split)
    for row in records:
        row_split = str(row.get("split_name", row.get("split", ""))).strip().lower()
        if _is_test_split_name(row_split):
            raise ValueError("refusing to build solver feedback impact from test split")

    landmark_positive: dict[int, dict[str, float]] = defaultdict(lambda: {"support": 0.0, "count": 0.0})
    landmark_negative: dict[int, dict[str, float]] = defaultdict(lambda: {"risk": 0.0, "count": 0.0})
    view_positive: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "count": 0.0})
    view_negative: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"weight": 0.0, "count": 0.0})
    detector_positive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detector_negative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    neutral_count = 0

    for row in records:
        if _label_role(row) in NEUTRAL_LABEL_ROLES:
            neutral_count += 1
        gid = _id_int(row.get("gaussian_id", row.get("matched_gaussian_id")))
        if gid < 0:
            continue
        query_id = str(row.get("query_id", ""))
        image_id = str(row.get("image_id", row.get("source_view_id", "")))
        pos = _positive_weight(row)
        neg = _negative_weight(row)
        if pos > 0.0:
            landmark_positive[gid]["support"] += float(pos)
            landmark_positive[gid]["count"] += 1.0
            view_positive[(str(gid), image_id)]["weight"] += float(pos)
            view_positive[(str(gid), image_id)]["count"] += 1.0
            detector_positive[query_id].append(_detector_entry(row, gid, pos))
        if neg > 0.0:
            landmark_negative[gid]["risk"] += float(neg)
            landmark_negative[gid]["count"] += 1.0
            view_negative[(str(gid), image_id)]["weight"] += float(neg)
            view_negative[(str(gid), image_id)]["count"] += 1.0
            detector_negative[query_id].append(_detector_entry(row, gid, neg))

    return {
        "schema_version": "solver_feedback_impact_v1",
        "split_name": split,
        "record_count": len(records),
        "landmark_positive": dict(landmark_positive),
        "landmark_negative": dict(landmark_negative),
        "view_positive": dict(view_positive),
        "view_negative": dict(view_negative),
        "detector_positive": dict(detector_positive),
        "detector_negative": dict(detector_negative),
        "query_features": payload.get("query_features", payload.get("query_global_features", {})),
        "metadata": {
            "positive_landmark_count": len(landmark_positive),
            "negative_landmark_count": len(landmark_negative),
            "positive_view_count": len(view_positive),
            "negative_view_count": len(view_negative),
            "positive_detector_query_count": len(detector_positive),
            "negative_detector_query_count": len(detector_negative),
            "neutral_correspondence_count": int(neutral_count),
        },
    }
