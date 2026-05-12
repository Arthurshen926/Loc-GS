#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from loc_gs.localization.pose_metrics import pose_error_summary


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _summary_from_details(rows: list[dict[str, Any]], stage: str) -> dict[str, float]:
    te_values: list[float] = []
    ae_values: list[float] = []
    inlier_values: list[int] = []
    for row in rows:
        te = _finite_float(row.get(f"{stage}_te"))
        ae = _finite_float(row.get(f"{stage}_ae"))
        if te is None or ae is None:
            continue
        te_values.append(te)
        ae_values.append(ae)
        inlier_values.append(int(_finite_float(row.get(f"{stage}_inliers")) or 0))
    return pose_error_summary(te_values, ae_values, inlier_values)


def _mean_matchability(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        matchability = row.get("matchability")
        if not isinstance(matchability, dict):
            continue
        for key, value in matchability.items():
            number = _finite_float(value)
            if number is None:
                continue
            buckets.setdefault(str(key), []).append(number)
    return {key: float(np.mean(values)) for key, values in sorted(buckets.items()) if values}


def _load_result_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("results", [])
    else:
        rows = data
    if not isinstance(rows, list):
        raise TypeError(f"results.json must contain a list or a dict with 'results': {path}")
    return [row for row in rows if isinstance(row, dict)]


def merge_eval_shards(shard_dirs: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    shards = [Path(path) for path in shard_dirs]
    if not shards:
        raise ValueError("at least one shard directory is required")

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for shard_dir in shards:
        summary_path = shard_dir / "summary.json"
        results_path = shard_dir / "results.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing shard summary: {summary_path}")
        if not results_path.exists():
            raise FileNotFoundError(f"missing shard results: {results_path}")
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        details.extend(_load_result_rows(results_path))

    details.sort(key=lambda row: str(row.get("image_name", "")))
    merged = dict(summaries[0])
    merged["query_offset"] = 0
    merged["query_stride"] = 1
    merged["query_shards"] = [
        {
            "dir": str(shard_dir),
            "query_offset": int(summary.get("query_offset", idx)),
            "query_stride": int(summary.get("query_stride", len(shards))),
            "queries": int(summary.get("queries", 0)),
            "localized": int(summary.get("localized", 0)),
        }
        for idx, (shard_dir, summary) in enumerate(zip(shards, summaries))
    ]
    merged["sparse"] = _summary_from_details(details, "sparse")
    merged["dense"] = _summary_from_details(details, "dense")
    merged["matchability"] = _mean_matchability(details)
    merged["localized"] = int(sum(1 for row in details if row.get("dense_te") is not None and row.get("dense_ae") is not None))
    merged["queries"] = len(details)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (output_path / "results.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return merged


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge query-sharded Cambridge hybrid eval outputs.")
    parser.add_argument("--shards", nargs="+", required=True, help="Shard eval directories containing summary/results JSON.")
    parser.add_argument("--output_dir", required=True, help="Merged eval output directory.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = merge_eval_shards(args.shards, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
