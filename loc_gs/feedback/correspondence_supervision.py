from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from loc_gs.feedback.schema import FeedbackMatchRecord


def _validate_split(split_name: str) -> str:
    split = str(split_name).strip()
    if not split:
        raise ValueError("split_name is required")
    if split.lower() == "test":
        raise ValueError("refusing to build correspondence supervision from test split")
    return split


def _gaussian_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value)
        if ":" in text:
            try:
                return int(text.rsplit(":", 1)[-1])
            except ValueError:
                return None
    return None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quality(record: FeedbackMatchRecord) -> float:
    descriptor = _float(record.descriptor_score, 0.0)
    margin = max(0.0, _float(record.descriptor_margin, 0.0))
    geometry = max(0.0, _float(record.local_geometry_score, 0.0))
    reproj = max(0.0, _float(record.reprojection_error_px, 4.0))
    reproj_weight = 1.0 / (1.0 + reproj)
    base = 0.35 * descriptor + 0.25 * min(margin, 1.0) + 0.25 * geometry + 0.15 * reproj_weight
    if record.pnp_inlier:
        base += 0.5
    return max(0.0, min(2.0, base))


def _record_to_row(
    record: FeedbackMatchRecord,
    index: int,
    *,
    query_regression_delta_cm: dict[str, float],
) -> dict[str, Any] | None:
    gid = _gaussian_id(record.matched_gaussian_id or record.matched_landmark_id)
    if gid is None:
        return None
    quality = _quality(record)
    label = 1 if bool(record.pnp_inlier) else 0
    query_id = str(record.query_id or record.image_id)
    delta_cm = max(0.0, float(query_regression_delta_cm.get(query_id, 0.0)))
    hard_negative = bool(label == 0 and delta_cm > 0.0)
    negative_multiplier = 1.0 + min(4.0, delta_cm / 20.0) if hard_negative else 1.0
    supervision_weight = float(quality * negative_multiplier)
    return {
        "index": int(index),
        "scene": record.scene,
        "query_id": query_id,
        "source_role": str(record.source_role or record.pose_source),
        "gaussian_id": int(gid),
        "keypoint_xy": list(record.keypoint_xy),
        "query_xy_norm": list(record.query_xy_norm) if record.query_xy_norm is not None else None,
        "image_cell": list(record.image_cell) if record.image_cell is not None else None,
        "depth_bin": record.depth_bin,
        "descriptor_score": _float(record.descriptor_score, 0.0),
        "descriptor_margin": _float(record.descriptor_margin, 0.0),
        "detector_score": _float(record.detector_score, 0.0),
        "reprojection_error_px": _float(record.reprojection_error_px, 999.0),
        "depth_m": _float(record.depth_m, 0.0),
        "local_geometry_score": _float(record.local_geometry_score, 0.0),
        "pnp_inlier": bool(record.pnp_inlier),
        "pose_success": bool(record.pose_success or record.pnp_success),
        "query_sparse_te_cm": _float(record.query_sparse_te_cm, _float(record.pose_error_t_cm, 0.0)),
        "label": label,
        "query_regression_delta_cm": float(delta_cm),
        "hard_negative": hard_negative,
        "negative_supervision_multiplier": float(negative_multiplier),
        "supervision_weight": supervision_weight,
    }


def build_correspondence_supervision(
    records: Iterable[FeedbackMatchRecord | dict[str, Any]],
    *,
    split_name: str,
    query_regression_delta_cm: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    split = _validate_split(split_name)
    regression_delta = {
        str(query_id): max(0.0, float(delta))
        for query_id, delta in (query_regression_delta_cm or {}).items()
        if _float(delta, 0.0) > 0.0
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        record = FeedbackMatchRecord.from_mapping(item) if isinstance(item, dict) else item
        row = _record_to_row(record, index, query_regression_delta_cm=regression_delta)
        if row is not None:
            rows.append(row)

    return {
        "schema_version": "correspondence_supervision_v1",
        "split_name": split,
        "record_count": int(len(rows)),
        "positive_count": int(sum(row["label"] for row in rows)),
        "hard_negative_count": int(sum(1 for row in rows if row.get("hard_negative"))),
        "query_regression_delta_cm": dict(sorted(regression_delta.items())),
        "records": rows,
    }


def export_supervision_targets(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = list(artifact.get("records", []))
    landmark_positive: dict[int, float] = defaultdict(float)
    landmark_total: dict[int, float] = defaultdict(float)
    detector_weights: list[float] = []
    labels: list[int] = []
    scorer_weights: list[float] = []
    by_query: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"pos": [], "neg": []})

    for row in rows:
        gid = int(row["gaussian_id"])
        weight = _float(row.get("supervision_weight"), 0.0)
        label = int(row.get("label", 0))
        landmark_total[gid] += max(weight, 1e-6)
        if label:
            landmark_positive[gid] += max(weight, 1e-6)
        detector_weights.append(weight if label else max(0.05, weight * 0.25))
        labels.append(label)
        scorer_weights.append(weight)
        bucket = "pos" if label else "neg"
        by_query[str(row.get("query_id", ""))][bucket].append(row)

    landmark_weights = {
        gid: landmark_positive.get(gid, 0.0) / max(total, 1e-6)
        for gid, total in sorted(landmark_total.items())
    }

    preferences: list[dict[str, Any]] = []
    conflict_edges: list[dict[str, Any]] = []
    for query_id, groups in sorted(by_query.items()):
        positives = sorted(groups["pos"], key=lambda row: -_float(row.get("supervision_weight")))
        negatives = sorted(groups["neg"], key=lambda row: -_float(row.get("descriptor_score")))
        if positives and negatives:
            pos = positives[0]
            neg = negatives[0]
            preferences.append(
                {
                    "query_id": query_id,
                    "positive_gaussian_id": int(pos["gaussian_id"]),
                    "negative_gaussian_id": int(neg["gaussian_id"]),
                    "margin": max(0.05, _float(pos.get("supervision_weight")) - _float(neg.get("supervision_weight"))),
                }
            )
        for pos in positives[:8]:
            for neg in negatives[:8]:
                descriptor_close = abs(_float(pos.get("descriptor_score")) - _float(neg.get("descriptor_score"))) <= 0.15
                low_margin_negative = _float(neg.get("descriptor_margin")) <= 0.05
                if descriptor_close or low_margin_negative:
                    hard_negative_multiplier = max(1.0, _float(neg.get("negative_supervision_multiplier"), 1.0))
                    conflict_edges.append(
                        {
                            "query_id": query_id,
                            "src": int(pos["gaussian_id"]),
                            "dst": int(neg["gaussian_id"]),
                            "weight": round(
                                (
                                    1.0
                                    + max(0.0, _float(neg.get("descriptor_score")) - _float(pos.get("descriptor_score")))
                                    + max(0.0, 0.1 - _float(neg.get("descriptor_margin")))
                                )
                                * hard_negative_multiplier,
                                6,
                            ),
                        }
                    )

    return {
        "schema_version": "correspondence_supervision_targets_v1",
        "descriptor_fusion": {
            "landmark_weights": landmark_weights,
            "role": "weight multi-view descriptor fusion toward PnP-inlier correspondences",
        },
        "detector_target": {
            "keypoint_weights": detector_weights,
            "records": [
                {
                    "query_id": row.get("query_id"),
                    "keypoint_xy": row.get("keypoint_xy"),
                    "query_xy_norm": row.get("query_xy_norm"),
                    "weight": detector_weights[idx],
                }
                for idx, row in enumerate(rows)
            ],
        },
        "match_scorer": {
            "labels": labels,
            "weights": scorer_weights,
            "gaussian_ids": [int(row["gaussian_id"]) for row in rows],
            "hard_negative": [bool(row.get("hard_negative", False)) for row in rows],
            "query_regression_delta_cm": [_float(row.get("query_regression_delta_cm"), 0.0) for row in rows],
            "query_xy_norm": [row.get("query_xy_norm") for row in rows],
            "image_cell": [row.get("image_cell") for row in rows],
            "depth_bin": [row.get("depth_bin") for row in rows],
            "detector_score": [_float(row.get("detector_score"), 0.0) for row in rows],
            "descriptor_score": [_float(row.get("descriptor_score"), 0.0) for row in rows],
            "descriptor_margin": [_float(row.get("descriptor_margin"), 0.0) for row in rows],
            "local_geometry_score": [_float(row.get("local_geometry_score"), 0.0) for row in rows],
        },
        "pnp_ranking": {
            "pairwise_preferences": preferences,
            "role": "rank PnP-useful inlier correspondences above high-score non-inliers",
        },
        "conflict_graph": {
            "edges": conflict_edges,
            "role": "penalize descriptor-similar but solver-inconsistent landmark pairs",
        },
    }
