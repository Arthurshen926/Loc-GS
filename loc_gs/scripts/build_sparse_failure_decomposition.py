#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loc_gs.diagnostics.sparse_failure_decomposition import decompose_sparse_failure
from loc_gs.feedback.io import load_feedback_bank
from loc_gs.feedback.schema import FeedbackMatchRecord


def _git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(cwd), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_from_result(item: dict[str, Any], index: int, scene: str) -> dict[str, Any]:
    sparse = item.get("sparse", {}) if isinstance(item.get("sparse"), dict) else {}
    dense = item.get("dense", [])
    final_dense = dense[-1] if isinstance(dense, list) and dense and isinstance(dense[-1], dict) else {}
    dense_stats = final_dense.get("dense_stats", {}) if isinstance(final_dense.get("dense_stats"), dict) else {}
    inliers = _as_float(sparse.get("inliers", sparse.get("num_inliers")))
    inlier_rate = _as_float(sparse.get("inlier_rate"))
    match_count = inliers / max(inlier_rate, 1e-6) if inlier_rate > 0.0 else 0.0
    return {
        "scene": scene,
        "query_id": str(item.get("query_id", item.get("image_id", f"query_{index:06d}"))),
        "match_count": match_count,
        "pnp_inlier_count": inliers,
        "all_match_inlier_rate": inlier_rate,
        "median_te_cm": _as_float(item.get("sparse_TE", item.get("sparse_te_cm"))),
        "render_valid_ratio": _as_float(dense_stats.get("valid_dense_match_ratio"), 1.0),
        "artifact_risk": _as_float(dense_stats.get("artifact_depth_risk_score"), 0.0),
        "render_feature_cosine_median": _as_float(dense_stats.get("render_feature_cosine_median"), 1.0),
    }


def _median(values: list[float], default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) * 0.5)


def _rows_from_feedback_bank(path: Path, scene: str) -> list[dict[str, Any]]:
    bank = load_feedback_bank(path)
    manifest = bank.get("manifest", {})
    split = str(manifest.get("split_name", manifest.get("split", ""))).strip().lower()
    if split == "test":
        raise ValueError("refusing to mine sparse failure decomposition from a test feedback bank")
    grouped: dict[str, list[FeedbackMatchRecord]] = {}
    for raw in bank.get("records", []):
        record = FeedbackMatchRecord.from_mapping(raw)
        query_id = record.query_id or record.image_id or "unknown"
        grouped.setdefault(query_id, []).append(record)

    rows: list[dict[str, Any]] = []
    for query_id, records in sorted(grouped.items()):
        count = len(records)
        inliers = sum(1 for record in records if record.pnp_inlier)
        scored = sorted(records, key=lambda record: _as_float(record.descriptor_score), reverse=True)
        top = scored[: min(50, len(scored))]
        top_rate = (sum(1 for record in top if record.pnp_inlier) / len(top)) if top else 0.0
        all_rate = inliers / count if count else 0.0
        margins = [_as_float(record.descriptor_margin) for record in records if record.descriptor_margin is not None]
        depths = [_as_float(record.depth_m) for record in records if record.depth_m is not None]
        positives = [record for record in records if record.pnp_inlier]
        positive_scores = [_as_float(record.descriptor_score) for record in positives if record.descriptor_score is not None]
        positive_score_median = _median(positive_scores, default=1.0)
        hard_score_non_inliers = [
            record
            for record in records
            if (not record.pnp_inlier) and _as_float(record.descriptor_score) >= positive_score_median * 0.95
        ]
        xy_cells = set()
        for record in records:
            if record.query_xy_norm is None:
                continue
            x = max(0, min(3, int(float(record.query_xy_norm[0]) * 4.0)))
            y = max(0, min(3, int(float(record.query_xy_norm[1]) * 4.0)))
            xy_cells.add((x, y))
        rows.append(
            {
                "scene": scene,
                "query_id": query_id,
                "match_count": count,
                "pnp_inlier_count": inliers,
                "all_match_inlier_rate": all_rate,
                "top_rank_inlier_rate": top_rate,
                "oracle_inlier_count": inliers,
                "image_coverage": len(xy_cells) / 16.0 if xy_cells else 1.0,
                "descriptor_margin_median": _median(margins, default=1.0),
                "descriptor_conflict_rate": len(hard_score_non_inliers) / max(count - inliers, 1),
                "depth_spread_m": (max(depths) - min(depths)) if len(depths) >= 2 else 1.0,
                "median_te_cm": _median(
                    [_as_float(record.pose_error_t_cm) for record in records if record.pose_error_t_cm is not None],
                    default=0.0,
                ),
            }
        )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-query sparse failure decomposition from ULF results.json.")
    parser.add_argument("--results_json", default=None, type=Path)
    parser.add_argument("--feedback_bank", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    if args.results_json is None and args.feedback_bank is None:
        raise ValueError("one of --results_json or --feedback_bank is required")
    if args.feedback_bank is not None:
        rows = _rows_from_feedback_bank(Path(args.feedback_bank), args.scene)
    else:
        payload = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("results_json must contain a list of per-query result objects")
        rows = [_row_from_result(item, idx, args.scene) for idx, item in enumerate(payload) if isinstance(item, dict)]
    reports = [decompose_sparse_failure(row) for row in rows]
    counts = Counter(report["primary_cause"] for report in reports)
    metrics = {
        "schema_version": "sparse_failure_decomposition_metrics_v1",
        "scene": str(args.scene),
        "split_name": str(args.split_name),
        "query_count": int(len(reports)),
        "cause_counts": dict(sorted(counts.items())),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "decomposition.json",
        {
            "schema_version": "sparse_failure_decomposition_report_v1",
            "scene": str(args.scene),
            "split_name": str(args.split_name),
            "queries": reports,
        },
    )
    with (output_dir / "decomposition.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "primary_cause", "active_flags"])
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "query_id": report["query_id"],
                    "primary_cause": report["primary_cause"],
                    "active_flags": ";".join(report["active_flags"]),
                }
            )
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(
        output_dir / "split_audit.json",
        {
            "schema_version": "sparse_failure_decomposition_split_audit_v1",
            "split_name": str(args.split_name),
            "test_split_used": str(args.split_name).strip().lower() == "test",
            "diagnostic_only": True,
        },
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "sparse_failure_decomposition_manifest_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "results_json": str(args.results_json) if args.results_json is not None else "",
            "feedback_bank": str(args.feedback_bank) if args.feedback_bank is not None else "",
            "output_dir": str(output_dir),
        },
    )
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path.cwd()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
