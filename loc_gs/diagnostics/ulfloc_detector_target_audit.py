from __future__ import annotations

from statistics import mean
from typing import Any, Mapping

import torch


def _points(entry: Mapping[str, Any], key: str) -> torch.Tensor:
    return torch.as_tensor(entry.get(key, []), dtype=torch.float32).reshape(-1, 2)


def _weights(entry: Mapping[str, Any], key: str, count: int) -> torch.Tensor:
    weights = torch.as_tensor(entry.get(key, []), dtype=torch.float32).reshape(-1)
    if weights.numel() == 0 and int(count) > 0:
        return torch.ones(int(count), dtype=torch.float32)
    if weights.numel() != int(count):
        raise ValueError(f"{key} must match point count")
    return weights


def summarize_detector_target_entry(
    entry: Mapping[str, Any],
    *,
    height: int,
    width: int,
    grid_size: int = 8,
) -> dict[str, float | int]:
    positive_yx = _points(entry, "keypoint_yx")
    negative_yx = _points(entry, "negative_keypoint_yx")
    weights = _weights(entry, "support_weights", int(positive_yx.shape[0]))
    bins = max(1, int(grid_size))
    h = max(1.0, float(height))
    w = max(1.0, float(width))
    cells: set[tuple[int, int]] = set()
    for y, x in positive_yx.tolist():
        cells.add(
            (
                min(bins - 1, max(0, int((float(y) / h) * bins))),
                min(bins - 1, max(0, int((float(x) / w) * bins))),
            )
        )
    if weights.numel() > 0:
        weight_mean = float(weights.mean().item())
        weight_std = float(weights.std(unbiased=False).item())
        weight_max = float(weights.max().item())
    else:
        weight_mean = 0.0
        weight_std = 0.0
        weight_max = 0.0
    if positive_yx.numel() > 0:
        centroid_y = float(positive_yx[:, 0].mean().item())
        centroid_x = float(positive_yx[:, 1].mean().item())
    else:
        centroid_y = float("nan")
        centroid_x = float("nan")
    return {
        "positive_count": int(positive_yx.shape[0]),
        "negative_count": int(negative_yx.shape[0]),
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "weight_max": weight_max,
        "coverage_cell_count": int(len(cells)),
        "coverage_fraction": float(len(cells) / float(bins * bins)),
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
    }


def sparse_result_error_map(payload: Mapping[str, Any]) -> dict[str, float]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise TypeError("sparse result payload must contain a rows list")
    errors: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        query_id = str(row.get("image_name", row.get("query_id", ""))).strip()
        if not query_id:
            continue
        value = row.get("sparse_te_cm", row.get("translation_error_cm", row.get("te_cm")))
        if value is None:
            continue
        errors[query_id] = float(value)
    return errors


def compare_detector_target_maps(
    baseline_targets: Mapping[str, Mapping[str, Any]],
    candidate_targets: Mapping[str, Mapping[str, Any]],
    *,
    height: int,
    width: int,
    grid_size: int = 8,
    baseline_errors: Mapping[str, float] | None = None,
    candidate_errors: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_errors = {} if baseline_errors is None else dict(baseline_errors)
    candidate_errors = {} if candidate_errors is None else dict(candidate_errors)
    rows: list[dict[str, Any]] = []
    for query_id in sorted(set(baseline_targets) & set(candidate_targets)):
        base = summarize_detector_target_entry(
            baseline_targets[query_id],
            height=int(height),
            width=int(width),
            grid_size=int(grid_size),
        )
        cand = summarize_detector_target_entry(
            candidate_targets[query_id],
            height=int(height),
            width=int(width),
            grid_size=int(grid_size),
        )
        row: dict[str, Any] = {"query_id": str(query_id)}
        for key, value in base.items():
            row[f"baseline_{key}"] = value
        for key, value in cand.items():
            row[f"candidate_{key}"] = value
        row["positive_count_delta"] = int(cand["positive_count"]) - int(base["positive_count"])
        row["negative_count_delta"] = int(cand["negative_count"]) - int(base["negative_count"])
        row["coverage_cell_delta"] = int(cand["coverage_cell_count"]) - int(base["coverage_cell_count"])
        row["weight_mean_delta"] = float(cand["weight_mean"]) - float(base["weight_mean"])
        if query_id in baseline_errors and query_id in candidate_errors:
            row["baseline_sparse_te_cm"] = float(baseline_errors[query_id])
            row["candidate_sparse_te_cm"] = float(candidate_errors[query_id])
            row["sparse_te_delta_cm"] = float(candidate_errors[query_id]) - float(baseline_errors[query_id])
        rows.append(row)

    deltas = [float(row["sparse_te_delta_cm"]) for row in rows if "sparse_te_delta_cm" in row]
    aggregate = {
        "common_query_count": int(len(rows)),
        "baseline_only_query_count": int(len(set(baseline_targets) - set(candidate_targets))),
        "candidate_missing_query_count": int(len(set(baseline_targets) - set(candidate_targets))),
        "candidate_only_query_count": int(len(set(candidate_targets) - set(baseline_targets))),
        "mean_positive_count_delta": float(mean([float(row["positive_count_delta"]) for row in rows])) if rows else 0.0,
        "mean_coverage_cell_delta": float(mean([float(row["coverage_cell_delta"]) for row in rows])) if rows else 0.0,
        "mean_weight_mean_delta": float(mean([float(row["weight_mean_delta"]) for row in rows])) if rows else 0.0,
        "mean_sparse_te_delta_cm": float(mean(deltas)) if deltas else None,
    }
    return rows, aggregate
