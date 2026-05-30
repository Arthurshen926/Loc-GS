from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PHASE0_TYPES = (
    "A_sparse_good_dense_bad",
    "B_sparse_marginal_dense_recoverable",
    "C_sparse_catastrophic",
    "D_normal_dense_good",
)


@dataclass(frozen=True)
class Phase0Thresholds:
    sparse_good_te_cm: float = 30.0
    dense_bad_margin_cm: float = 20.0
    sparse_marginal_te_cm: float = 200.0
    dense_recovery_margin_cm: float = 20.0
    sparse_catastrophic_te_cm: float = 300.0
    normal_dense_te_cm: float = 15.0
    normal_sparse_te_cm: float = 50.0


def _float(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _optional_float(row: Mapping[str, Any], key: str) -> float | None:
    value = _float(row, key)
    return value if np.isfinite(value) else None


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def classify_phase0_case(row: Mapping[str, Any], thresholds: Phase0Thresholds = Phase0Thresholds()) -> str | None:
    """Classify a train/self-map query into sparse-to-dense transition types."""

    sparse = _float(row, "sparse_te_cm")
    dense = _float(row, "base_dense_te_cm")
    if not np.isfinite(sparse) or not np.isfinite(dense):
        return None
    if sparse >= float(thresholds.sparse_catastrophic_te_cm):
        return "C_sparse_catastrophic"
    if dense <= float(thresholds.normal_dense_te_cm) and sparse <= float(thresholds.normal_sparse_te_cm):
        return "D_normal_dense_good"
    if sparse <= float(thresholds.sparse_good_te_cm) and dense - sparse >= float(thresholds.dense_bad_margin_cm):
        return "A_sparse_good_dense_bad"
    if (
        float(thresholds.sparse_good_te_cm) < sparse <= float(thresholds.sparse_marginal_te_cm)
        and sparse - dense >= float(thresholds.dense_recovery_margin_cm)
    ):
        return "B_sparse_marginal_dense_recoverable"
    return None


def normalize_phase0_row(
    row: Mapping[str, Any],
    *,
    case_type: str,
    source_path: str = "",
    paper_safe_for_tuning: bool = True,
) -> dict[str, Any]:
    sparse_te = _float(row, "sparse_te_cm")
    base_dense_te = _float(row, "base_dense_te_cm")
    sparse_conditioned_te = _optional_float(row, "sparse_conditioned_dense_te_cm")
    return {
        "scene": str(row.get("scene", "")),
        "query_index": _int(row, "query_index"),
        "image_name": str(row.get("image_name", "")),
        "source_split": str(row.get("split", "unknown")),
        "paper_safe_for_tuning": bool(paper_safe_for_tuning),
        "case_type": str(case_type),
        "sparse_te_cm": sparse_te,
        "base_dense_te_cm": base_dense_te,
        "sparse_conditioned_dense_te_cm": sparse_conditioned_te,
        "delta_dense_minus_sparse_cm": base_dense_te - sparse_te,
        "sparse_conditioned_delta_cm": (
            None if sparse_conditioned_te is None else sparse_conditioned_te - base_dense_te
        ),
        "sparse_inlier_count": _int(row, "sparse_inlier_count"),
        "base_dense_inlier_count": _int(row, "base_dense_inlier_count"),
        "sparse_conditioned_label": str(row.get("sparse_conditioned_label", "")),
        "sparse_conditioned_decision": str(row.get("sparse_conditioned_decision", "")),
        "transition_decision": str(row.get("transition_decision", "")),
        "source_path": str(row.get("_source_path", source_path)),
    }


def _sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row.get("scene", "")), _int(row, "query_index"), str(row.get("image_name", ""))


def make_phase0_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Phase0Thresholds = Phase0Thresholds(),
    max_per_type: int = 50,
    val_fraction: float = 0.25,
    source_path: str = "",
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {case_type: [] for case_type in PHASE0_TYPES}
    for row in rows:
        case_type = classify_phase0_case(row, thresholds)
        if case_type is None:
            continue
        source_split = str(row.get("split", "unknown"))
        paper_safe = source_split == "train"
        buckets[case_type].append(
            normalize_phase0_row(
                row,
                case_type=case_type,
                source_path=source_path,
                paper_safe_for_tuning=paper_safe,
            )
        )

    selected: list[dict[str, Any]] = []
    for case_type in PHASE0_TYPES:
        bucket = sorted(buckets[case_type], key=_sort_key)
        if int(max_per_type) > 0:
            bucket = bucket[: int(max_per_type)]
        val_count = int(round(len(bucket) * float(val_fraction)))
        for idx, item in enumerate(bucket):
            split = "val" if idx < val_count else "train"
            selected.append({**item, "phase0_split": split})
    selected = sorted(selected, key=lambda row: (str(row["case_type"]), str(row["phase0_split"]), _sort_key(row)))
    type_counts = {case_type: sum(1 for row in selected if row["case_type"] == case_type) for case_type in PHASE0_TYPES}
    split_counts = {
        "train": sum(1 for row in selected if row["phase0_split"] == "train"),
        "val": sum(1 for row in selected if row["phase0_split"] == "val"),
    }
    return {
        "schema": "loc_gs_clean_render_phase0_benchmark_v1",
        "diagnostic_only": True,
        "cases": selected,
        "summary": {
            "total_selected": int(len(selected)),
            "type_counts": type_counts,
            "split_counts": split_counts,
            "thresholds": {
                "sparse_good_te_cm": float(thresholds.sparse_good_te_cm),
                "dense_bad_margin_cm": float(thresholds.dense_bad_margin_cm),
                "sparse_marginal_te_cm": float(thresholds.sparse_marginal_te_cm),
                "dense_recovery_margin_cm": float(thresholds.dense_recovery_margin_cm),
                "sparse_catastrophic_te_cm": float(thresholds.sparse_catastrophic_te_cm),
                "normal_dense_te_cm": float(thresholds.normal_dense_te_cm),
                "normal_sparse_te_cm": float(thresholds.normal_sparse_te_cm),
            },
        },
    }
