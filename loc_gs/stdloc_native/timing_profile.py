from __future__ import annotations

from typing import Any, Iterable

import torch


TIMING_FIELDS = (
    "feature",
    "sparse_total",
    "sparse_pose",
    "sparse_match",
    "dense_total",
    "dense_pose",
    "dense_match_render",
    "total",
)

POSE_RELIABILITY_FIELDS = (
    "match_count",
    "inlier_count",
    "inlier_ratio",
    "all_reprojection_mean_px",
    "all_reprojection_median_px",
    "all_reprojection_p90_px",
    "inlier_reprojection_mean_px",
    "inlier_reprojection_median_px",
    "inlier_reprojection_p90_px",
)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def finalize_query_timing(row: dict[str, Any]) -> dict[str, Any]:
    """Add derived per-query stage timings without mutating the input row."""

    out = dict(row)
    sparse_total = _as_float(out.get("sparse_total_ms"))
    sparse_pose = _as_float(out.get("sparse_pose_ms"))
    dense_total = _as_float(out.get("dense_total_ms"))
    dense_pose = _as_float(out.get("dense_pose_ms"))
    out["sparse_match_ms"] = max(0.0, sparse_total - sparse_pose)
    out["dense_match_render_ms"] = max(0.0, dense_total - dense_pose)
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _stage_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    key = f"{field}_ms"
    return [_as_float(row[key]) for row in rows if key in row]


def _stage_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": _mean(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


def _pose_reliability_values(rows: Iterable[dict[str, Any]], stage: str, field: str) -> list[float]:
    key = f"{stage}_pose_reliability"
    values: list[float] = []
    for row in rows:
        payload = row.get(key)
        if not isinstance(payload, dict) or field not in payload:
            continue
        value = payload[field]
        if value is None:
            continue
        values.append(_as_float(value))
    return values


def _pose_reliability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for stage in ("sparse", "dense"):
        stage_rows = [row for row in rows if isinstance(row.get(f"{stage}_pose_reliability"), dict)]
        if not stage_rows:
            continue
        payload[stage] = {
            "count": int(len(stage_rows)),
            **{
                field: _stage_summary(_pose_reliability_values(stage_rows, stage, field))
                for field in POSE_RELIABILITY_FIELDS
                if _pose_reliability_values(stage_rows, stage, field)
            },
        }
    return payload


def aggregate_timing_profile(
    rows: list[dict[str, Any]],
    *,
    scene: str,
    method: str,
    landmark_count: int | None = None,
    dense_iterations: int | None = None,
) -> dict[str, Any]:
    """Summarize query-level profiler rows into a paper-audit friendly payload."""

    finalized = [finalize_query_timing(row) for row in rows]
    latency = {
        field: _stage_summary(_stage_values(finalized, field))
        for field in TIMING_FIELDS
        if _stage_values(finalized, field)
    }
    total_mean = float(latency.get("total", {}).get("mean", 0.0))
    payload: dict[str, Any] = {
        "scene": str(scene),
        "method": str(method),
        "queries": int(len(finalized)),
        "latency_ms": latency,
        "fps": {
            "mean_latency": (1000.0 / total_mean) if total_mean > 0.0 else 0.0,
        },
        "per_query": finalized,
    }
    reliability = _pose_reliability_summary(finalized)
    if reliability:
        payload["pose_reliability"] = reliability
    if landmark_count is not None:
        payload["landmark_count"] = int(landmark_count)
    if dense_iterations is not None:
        payload["dense_iterations"] = int(dense_iterations)
    return payload
