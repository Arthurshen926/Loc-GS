#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from loc_gs.localization.pose_metrics import POSE_RECALL_THRESHOLDS, recall_metric_key
from loc_gs.scripts.select_cambridge_branch import (
    _branch_path,
    filter_rows,
    load_branch_manifest,
    load_calibration_ids,
    load_result_rows,
    result_dir_from_path,
    summarize_rows,
)


RECALL_REPORT_METRICS = (
    "median_te",
    "median_ae",
    *[key for key, _te_thr, _ae_thr in POSE_RECALL_THRESHOLDS],
    "avg_inliers",
)


def _metric_score(metrics: dict[str, float], target_metric: str) -> float:
    if target_metric == "median_te":
        return -float(metrics["median_te"])
    key = recall_metric_key(target_metric)
    if key not in metrics:
        raise KeyError(f"unknown recall metric: {target_metric}")
    return float(metrics[key])


def _macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {
        "mean_median_te": float(np.mean([row["median_te"] for row in rows])) if rows else float("inf"),
        "mean_median_ae": float(np.mean([row["median_ae"] for row in rows])) if rows else float("inf"),
        "queries": float(sum(row["queries"] for row in rows)),
        "localized": float(sum(row["localized"] for row in rows)),
    }
    for key, _te_thr, _ae_thr in POSE_RECALL_THRESHOLDS:
        out[f"macro_{key}"] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    return out


def collect_branch_metrics(
    manifest: dict[str, dict[str, dict[str, Any]]],
    *,
    stage: str = "dense",
    calibration_ids: set[str] | None = None,
    calibration_stride: int = 0,
    calibration_offset: int = 0,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    report: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for scene, branches in manifest.items():
        report[scene] = {}
        for branch, spec in branches.items():
            report[scene][branch] = {}
            for split, path_key in (("calibration", "calibration_dir"), ("test", "test_dir")):
                result_dir = _branch_path(spec, path_key)
                rows = load_result_rows(result_dir, stage=stage)
                if split == "calibration":
                    rows = filter_rows(
                        rows,
                        ids=calibration_ids,
                        stride=calibration_stride,
                        offset=calibration_offset,
                    )
                metrics = summarize_rows(rows)
                report[scene][branch][split] = {
                    **metrics,
                    "result_dir": str(result_dir_from_path(result_dir)),
                }
    return report


def oracle_by_metric(
    branch_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    split: str,
    target_metric: str,
) -> dict[str, Any]:
    selected: dict[str, str] = {}
    selected_rows: list[dict[str, Any]] = []
    for scene, branches in branch_metrics.items():
        best_branch = ""
        best_score = -math.inf
        best_metrics: dict[str, Any] | None = None
        for branch, split_metrics in branches.items():
            metrics = split_metrics[split]
            score = _metric_score(metrics, target_metric)
            if score > best_score:
                best_branch = branch
                best_score = score
                best_metrics = metrics
        if best_metrics is None:
            raise ValueError(f"no branch metrics for scene {scene}")
        selected[scene] = best_branch
        selected_rows.append({"scene": scene, "selected_branch": best_branch, **best_metrics})
    return {
        "split": split,
        "target_metric": target_metric,
        "selected_branch": selected,
        "macro": _macro(selected_rows),
        "rows": selected_rows,
    }


def write_branch_metrics_csv(
    branch_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "branch", "split", *RECALL_REPORT_METRICS, "localized", "queries", "result_dir"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene, branches in branch_metrics.items():
            for branch, split_metrics in branches.items():
                for split, metrics in split_metrics.items():
                    writer.writerow(
                        {
                            "scene": scene,
                            "branch": branch,
                            "split": split,
                            **{key: metrics.get(key, 0.0) for key in RECALL_REPORT_METRICS},
                            "localized": int(metrics["localized"]),
                            "queries": int(metrics["queries"]),
                            "result_dir": metrics["result_dir"],
                        }
                    )


def write_oracle_csv(oracles: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recall_fields = [key for key, _te_thr, _ae_thr in POSE_RECALL_THRESHOLDS]
    fieldnames = [
        "split",
        "target_metric",
        "scene",
        "selected_branch",
        "median_te",
        "median_ae",
        *recall_fields,
        "localized",
        "queries",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for oracle in oracles:
            for row in oracle["rows"]:
                writer.writerow(
                    {
                        "split": oracle["split"],
                        "target_metric": oracle["target_metric"],
                        "scene": row["scene"],
                        "selected_branch": row["selected_branch"],
                        "median_te": row["median_te"],
                        "median_ae": row["median_ae"],
                        **{key: row.get(key, 0.0) for key in recall_fields},
                        "localized": int(row["localized"]),
                        "queries": int(row["queries"]),
                    }
                )
            macro = oracle["macro"]
            writer.writerow(
                {
                    "split": oracle["split"],
                    "target_metric": oracle["target_metric"],
                    "scene": "MACRO_MEAN",
                    "selected_branch": "",
                    "median_te": macro["mean_median_te"],
                    "median_ae": macro["mean_median_ae"],
                    **{key: macro[f"macro_{key}"] for key in recall_fields},
                    "localized": int(macro["localized"]),
                    "queries": int(macro["queries"]),
                }
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report Cambridge branch pose-recall headroom.")
    parser.add_argument("--manifest", required=True, help="JSON mapping scene -> branch -> result directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--calibration_ids", default="", help="Optional txt/json calibration image ids.")
    parser.add_argument("--calibration_stride", type=int, default=0)
    parser.add_argument("--calibration_offset", type=int, default=0)
    parser.add_argument(
        "--oracle_metrics",
        nargs="+",
        default=["median_te", "r10", "r5", "r2"],
        help="Metrics used to estimate per-scene oracle branch headroom.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest = load_branch_manifest(args.manifest)
    calibration_ids = load_calibration_ids(args.calibration_ids) if args.calibration_ids else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_metrics = collect_branch_metrics(
        manifest,
        stage=args.stage,
        calibration_ids=calibration_ids,
        calibration_stride=args.calibration_stride,
        calibration_offset=args.calibration_offset,
    )
    oracles = [
        oracle_by_metric(branch_metrics, split=split, target_metric=metric)
        for split in ("calibration", "test")
        for metric in args.oracle_metrics
    ]
    report = {
        "stage": args.stage,
        "manifest": str(args.manifest),
        "branch_metrics": branch_metrics,
        "oracles": oracles,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_branch_metrics_csv(branch_metrics, output_dir / "branch_metrics.csv")
    write_oracle_csv(oracles, output_dir / "oracle_headroom.csv")
    print(json.dumps({"oracles": [{k: v for k, v in oracle.items() if k != "rows"} for oracle in oracles]}, indent=2))


if __name__ == "__main__":
    main()
