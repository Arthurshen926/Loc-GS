from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path_or_dir: str | Path, filename: str) -> Any:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stage_summary(stage: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "median_te_cm": ("median_te_cm", "median_te"),
        "median_re_deg": ("median_re_deg", "median_ae", "median_re"),
        "recall_5m_10deg": ("recall_5m_10d", "recall_5m_10deg"),
        "recall_2m_5deg": ("recall_2m_5d", "recall_2m_5deg"),
        "recall_10cm_5deg": ("recall_10cm_5deg", "recall_10cm_5d"),
        "recall_5cm_5deg": ("recall_5cm_5d", "recall_5cm_5deg"),
        "recall_2cm_2deg": ("recall_2cm_2d", "recall_2cm_2deg"),
        "avg_inliers": ("avg_inliers", "mean_inliers"),
    }
    out: dict[str, float] = {}
    for key, names in aliases.items():
        for name in names:
            value = _as_float(stage.get(name))
            if value is not None:
                out[key] = value
                break
    return out


def load_stdloc_summary(path_or_dir: str | Path) -> dict[str, Any]:
    data = _load_json(path_or_dir, "summary.json")
    if not isinstance(data, dict):
        raise ValueError("STDLoc summary must be a JSON object")
    out = {key: value for key, value in data.items() if key not in {"sparse", "dense"}}
    out["sparse"] = _stage_summary(data.get("sparse", {}))
    out["dense"] = _stage_summary(data.get("dense", {}))
    return out


def _query_id(row: dict[str, Any], fallback_index: int | None = None) -> str:
    for key in ("query_id", "image_name", "name", "query", "image"):
        value = row.get(key)
        if value is not None:
            return str(value)
    if fallback_index is not None:
        return f"query_{int(fallback_index):06d}"
    raise ValueError(f"query result row has no query identifier: {row}")


def _stage_inliers(row: dict[str, Any], stage: str) -> int:
    value = row.get(f"{stage}_inliers")
    if value is not None:
        return int(value)
    nested = row.get(stage)
    if isinstance(nested, dict):
        return int(nested.get("inliers", 0))
    if isinstance(nested, list) and nested:
        last = nested[-1]
        if isinstance(last, dict):
            return int(last.get("inliers", 0))
    return 0


def _stage_metric(row: dict[str, Any], stage: str, *names: str) -> float | None:
    for name in names:
        value = _as_float(row.get(name))
        if value is not None:
            return value
    nested = row.get(stage)
    if isinstance(nested, dict):
        for name in names:
            value = _as_float(nested.get(name))
            if value is not None:
                return value
    return None


def load_stdloc_query_results(path_or_dir: str | Path) -> list[dict[str, Any]]:
    data = _load_json(path_or_dir, "results.json")
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError("STDLoc results must be a JSON list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "query_id": _query_id(row, fallback_index=index),
                "sparse_re_deg": _stage_metric(row, "sparse", "sparse_AE", "sparse_ae", "sparse_re_deg"),
                "sparse_te_cm": _stage_metric(row, "sparse", "sparse_TE", "sparse_te", "sparse_te_cm"),
                "sparse_inliers": _stage_inliers(row, "sparse"),
                "dense_re_deg": _stage_metric(row, "dense", "dense_AE", "dense_ae", "dense_re_deg"),
                "dense_te_cm": _stage_metric(row, "dense", "dense_TE", "dense_te", "dense_te_cm"),
                "dense_inliers": _stage_inliers(row, "dense"),
            }
        )
    return rows
