from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split


def build_sparse_metrics_from_results(
    *,
    results_path: str | Path,
    scene: str,
    split_name: str,
    query_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = reject_test_split(split_name, purpose="internal sparse results metrics")
    source_rows = _load_result_rows(results_path)
    requested = [str(query_id) for query_id in query_ids] if query_ids is not None else None
    requested_set = set(requested) if requested is not None else None
    rows: list[dict[str, object]] = []
    source_by_query: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        query_id = _query_id(row)
        source_by_query[query_id] = row
        if requested_set is not None and query_id not in requested_set:
            continue
        te_cm = _float_from(row, "te_cm", "sparse_te_cm")
        re_deg = _float_from(row, "re_deg", "sparse_re_deg")
        inliers = _int_from(row, "inlier_count", "sparse_inliers")
        rows.append(
            {
                "query_id": query_id,
                "te_cm": te_cm,
                "re_deg": re_deg,
                "inlier_count": inliers,
            }
        )
    missing = [query_id for query_id in requested or [] if query_id not in source_by_query]
    te_values = [float(row["te_cm"]) for row in rows if row["te_cm"] is not None]
    re_values = [float(row["re_deg"]) for row in rows if row["re_deg"] is not None]
    inlier_values = [int(row["inlier_count"]) for row in rows if row["inlier_count"] is not None]
    summary = {
        "schema_version": "internal_sparse_results_metrics_v1",
        "scene": str(scene),
        "split_name": split,
        "source_results": str(results_path),
        "query_filter_enabled": bool(requested is not None),
        "requested_query_count": None if requested is None else int(len(requested)),
        "matched_query_count": None if requested is None else int(len(rows)),
        "missing_query_count": None if requested is None else int(len(missing)),
        "missing_query_ids_preview": missing[:10],
        "query_count": int(len(rows)),
        "mean_inliers": float(mean(inlier_values)) if inlier_values else 0.0,
        "median_te_cm": _median_or_none(te_values),
        "median_re_deg": _median_or_none(re_values),
        "recall_10cm_5d": _recall(te_values, re_values, total_count=len(rows), te_threshold_cm=10.0, re_threshold_deg=5.0),
        "recall_5cm_5d": _recall(te_values, re_values, total_count=len(rows), te_threshold_cm=5.0, re_threshold_deg=5.0),
        "pose_metric_status": "verified" if te_values else "missing_pose_metrics",
    }
    return summary, rows


def load_query_ids(path: str | Path) -> list[str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict) and "query_ids" in payload:
            payload = payload["query_ids"]
        if not isinstance(payload, list):
            raise ValueError(f"query id JSON must be a list or contain query_ids: {path}")
        return [str(value) for value in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _load_result_rows(path: str | Path) -> list[Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"results file must contain a list or rows list: {path}")
    return [row for row in rows if isinstance(row, Mapping)]


def _query_id(row: Mapping[str, Any]) -> str:
    for key in ("query_id", "image_id", "image_name"):
        if key in row:
            return str(row[key])
    raise ValueError(f"result row is missing query id field: {row}")


def _float_from(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])
    return None


def _int_from(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in row and row[key] is not None:
            return int(row[key])
    return None


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _recall(
    te_values: Sequence[float],
    re_values: Sequence[float],
    *,
    total_count: int,
    te_threshold_cm: float,
    re_threshold_deg: float,
) -> float:
    if total_count <= 0:
        return 0.0
    passed = 0
    for te_cm, re_deg in zip(te_values, re_values):
        if float(te_cm) <= float(te_threshold_cm) and float(re_deg) <= float(re_threshold_deg):
            passed += 1
    return float(passed / total_count)
