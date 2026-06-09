from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from loc_gs.feedback.correspondence_supervision import export_supervision_targets


SCHEMA_VERSION = "ulfloc_solver_feedback_v1"
CORRESPONDENCE_SCHEMA_VERSION = "ulfloc_correspondence_feedback_v1"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_weight(row: dict[str, Any], *, positive: bool, boost_strength: float, deboost_strength: float) -> float:
    quality = max(0.0, _float(row.get("supervision_weight"), 0.0))
    quality = min(1.0, quality)
    if positive:
        return 1.0 + float(boost_strength) * quality
    descriptor_score = max(0.0, min(1.0, _float(row.get("descriptor_score"), 0.0)))
    margin = max(0.0, min(1.0, _float(row.get("descriptor_margin"), 0.0)))
    hard_negative = descriptor_score * (1.0 - margin)
    return 1.0 - float(deboost_strength) * hard_negative


def _average(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def build_ulfloc_correspondence_feedback(
    supervision: dict[str, Any],
    *,
    num_landmarks: int,
    boost_strength: float = 0.5,
    deboost_strength: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 1.75,
) -> dict[str, Any]:
    """Build a ULF-compatible solver feedback payload from correspondence labels.

    The returned dictionary can be pickled as ``solver_feedback.pkl`` and loaded
    by the existing ULF-Loc solver feedback hooks. Extra correspondence targets
    are carried alongside the native-compatible fields for detector/scorer
    training.
    """

    split_name = str(supervision.get("split_name", "")).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to build ULF correspondence feedback from test split")
    count = int(num_landmarks)
    if count <= 0:
        raise ValueError(f"num_landmarks must be positive, got {num_landmarks}")

    rows = list(supervision.get("records", []))
    per_landmark: dict[int, list[float]] = defaultdict(list)
    per_view: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        gid = int(row["gaussian_id"])
        if gid < 0 or gid >= count:
            raise ValueError(f"gaussian_id {gid} outside num_landmarks {count}")
        positive = int(row.get("label", 0)) > 0
        weight = _row_weight(
            row,
            positive=positive,
            boost_strength=boost_strength,
            deboost_strength=deboost_strength,
        )
        weight = max(float(min_weight), min(float(max_weight), weight))
        per_landmark[gid].append(weight)
        query_id = str(row.get("query_id", ""))
        if query_id:
            per_view[query_id][gid].append(weight)

    landmark_weights = torch.ones((count,), dtype=torch.float32)
    for gid, values in per_landmark.items():
        landmark_weights[int(gid)] = float(_average(values))
    landmark_weights = landmark_weights.clamp(float(min_weight), float(max_weight))

    view_landmark_weights: dict[str, dict[str, torch.Tensor]] = {}
    for query_id, values_by_gid in sorted(per_view.items()):
        ids = sorted(values_by_gid)
        if not ids:
            continue
        view_landmark_weights[query_id] = {
            "landmark_ids": torch.as_tensor(ids, dtype=torch.long),
            "weights": torch.as_tensor(
                [_average(values_by_gid[gid]) for gid in ids],
                dtype=torch.float32,
            ).clamp(float(min_weight), float(max_weight)),
        }

    targets = export_supervision_targets(supervision)
    positive_count = int(supervision.get("positive_count", 0))
    record_count = int(supervision.get("record_count", len(rows)))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "correspondence_schema_version": CORRESPONDENCE_SCHEMA_VERSION,
        "split_name": split_name or "unknown",
        "landmark_weights": landmark_weights,
        "view_landmark_weights": view_landmark_weights,
        "correspondence_targets": targets,
        "metadata": {
            "source_schema_version": str(supervision.get("schema_version", "")),
            "num_landmarks": count,
            "record_count": record_count,
            "positive_count": positive_count,
            "negative_count": int(record_count - positive_count),
            "boost_strength": float(boost_strength),
            "deboost_strength": float(deboost_strength),
            "min_weight": float(min_weight),
            "max_weight": float(max_weight),
            "detector_target_count": int(len(targets["detector_target"]["records"])),
            "ranking_preference_count": int(len(targets["pnp_ranking"]["pairwise_preferences"])),
            "conflict_edge_count": int(len(targets["conflict_graph"]["edges"])),
        },
    }
    return payload

