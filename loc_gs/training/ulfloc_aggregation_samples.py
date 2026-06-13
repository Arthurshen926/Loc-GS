from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


POSITIVE_LABELS = {"protected_support", "positive_inlier"}
NEGATIVE_LABELS = {"harmful_negative", "risky_competitor_negative"}
VALID_LABELS = POSITIVE_LABELS | NEGATIVE_LABELS | {"neutral_outlier"}


@dataclass(frozen=True)
class AggregationSample:
    gaussian_id: int
    sampled_row: int
    view_id: str
    query_id: str
    descriptor: torch.Tensor
    label_role: str
    source_role: str
    target: int
    protected: bool
    reliability: float


def _feedback_split_name(feedback_v4: Mapping[str, Any]) -> str:
    split = str(feedback_v4.get("split_name", "")).strip()
    audit = feedback_v4.get("split_audit", {})
    audit_test = bool(audit.get("test_split_used", False) or audit.get("official_test_used", False)) if isinstance(audit, Mapping) else False
    audit_split = str(audit.get("split_name", "")).strip() if isinstance(audit, Mapping) else ""
    if split.lower() == "test" or audit_test:
        raise ValueError("test split is not allowed for ULF aggregation samples")
    if audit_split.lower() == "test":
        raise ValueError("test split is not allowed for ULF aggregation samples")
    if audit_split and split and audit_split != split:
        raise ValueError(f"split_name mismatch: expected {split}, got audit split {audit_split}")
    if not split:
        raise ValueError("split_name is required")
    return split


def _descriptor(row: Mapping[str, Any]) -> torch.Tensor:
    if "query_descriptor" not in row:
        raise KeyError("feedback correspondence is missing query_descriptor")
    desc = torch.as_tensor(row["query_descriptor"], dtype=torch.float32).reshape(-1)
    if desc.numel() == 0:
        raise ValueError("query_descriptor must be non-empty")
    if not bool(torch.isfinite(desc).all()):
        raise ValueError("query_descriptor must be finite")
    return torch.nn.functional.normalize(desc, p=2, dim=0)


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if value == value else float(default)


def _reliability(row: Mapping[str, Any], *, target: int) -> float:
    score = max(0.0, _float(row, "descriptor_score", 0.0))
    margin = max(0.0, _float(row, "descriptor_margin", 0.0))
    reproj = max(0.0, _float(row, "reprojection_error_px", 0.0))
    detector = max(0.0, _float(row, "detector_score", 0.0))
    if target > 0:
        return float((1.0 + score + margin + detector) / (1.0 + reproj))
    return float(1.0 + score + margin + detector)


def build_aggregation_samples(
    feedback_v4: Mapping[str, Any],
    *,
    include_render_aug: bool = True,
) -> list[AggregationSample]:
    """Convert Feedback v4 correspondences into aggregation-stage view samples."""

    split = _feedback_split_name(feedback_v4)
    samples: list[AggregationSample] = []
    for raw in feedback_v4.get("correspondences", []):
        if not isinstance(raw, Mapping):
            continue
        row_split = str(raw.get("split_name", split)).strip()
        if row_split.lower() == "test":
            raise ValueError("test split is not allowed in aggregation sample rows")
        if row_split and row_split != split:
            raise ValueError(f"correspondence split_name mismatch: expected {split}, got {row_split}")
        source_role = str(raw.get("source_role", "")).strip()
        if source_role == "render_aug_trace" and not bool(include_render_aug):
            continue
        label_role = str(raw.get("label_role", "")).strip()
        if label_role == "neutral_outlier":
            continue
        if label_role not in VALID_LABELS:
            continue
        target = 1 if label_role in POSITIVE_LABELS else 0
        gaussian_id = int(raw["gaussian_id"])
        sampled_row = int(raw["sampled_row"])
        samples.append(
            AggregationSample(
                gaussian_id=gaussian_id,
                sampled_row=sampled_row,
                view_id=str(raw.get("image_id", raw.get("query_id", ""))),
                query_id=str(raw.get("query_id", "")),
                descriptor=_descriptor(raw),
                label_role=label_role,
                source_role=source_role,
                target=int(target),
                protected=label_role == "protected_support",
                reliability=_reliability(raw, target=target),
            )
        )
    return samples


def group_samples_by_landmark(samples: list[AggregationSample]) -> dict[int, list[AggregationSample]]:
    grouped: dict[int, list[AggregationSample]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample.gaussian_id)].append(sample)
    return dict(grouped)
