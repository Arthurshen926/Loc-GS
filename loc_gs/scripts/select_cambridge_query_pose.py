#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from loc_gs.scripts.select_cambridge_branch import (
    branch_score,
    filter_rows,
    load_branch_manifest,
    load_result_rows,
    result_dir_from_path,
    summarize_rows,
)


def _finite(value: float | None, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    value = float(value)
    return value if math.isfinite(value) else fallback


def _common_name_ratio(branch_rows: dict[str, list[dict[str, Any]]]) -> float:
    sets = []
    for rows in branch_rows.values():
        names = {str(row["image_name"]) for row in rows if not str(row["image_name"]).startswith("idx:")}
        if not names:
            return 0.0
        if names:
            sets.append(names)
    if len(sets) < 2:
        return 0.0
    common = set.intersection(*sets)
    denom = max(1, min(len(names) for names in sets))
    return len(common) / float(denom)


def align_query_rows(
    branch_rows: dict[str, list[dict[str, Any]]],
    align_by: str = "auto",
) -> list[dict[str, dict[str, Any]]]:
    if not branch_rows:
        return []
    mode = align_by
    if mode == "auto":
        mode = "name" if _common_name_ratio(branch_rows) >= 0.8 else "index"
    if mode == "name":
        common = None
        by_branch = {}
        for branch, rows in branch_rows.items():
            mapping = {str(row["image_name"]): row for row in rows}
            by_branch[branch] = mapping
            keys = set(mapping)
            common = keys if common is None else common & keys
        if not common:
            return []
        return [{branch: by_branch[branch][key] for branch in branch_rows} for key in sorted(common)]
    if mode != "index":
        raise ValueError(f"unsupported alignment mode: {align_by}")
    length = min(len(rows) for rows in branch_rows.values())
    return [{branch: rows[idx] for branch, rows in branch_rows.items()} for idx in range(length)]


def calibration_branch_deltas(
    branch_rows: dict[str, list[dict[str, Any]]],
    *,
    baseline_branch: str,
    metric: str,
    te_penalty_per_cm: float,
) -> dict[str, float]:
    metrics = {branch: summarize_rows(rows) for branch, rows in branch_rows.items()}
    baseline = metrics.get(baseline_branch)
    if baseline is None:
        return {branch: 0.0 for branch in branch_rows}
    baseline_score = branch_score(baseline, baseline, metric, te_penalty_per_cm)
    return {
        branch: branch_score(branch_metrics, baseline, metric, te_penalty_per_cm) - baseline_score
        for branch, branch_metrics in metrics.items()
    }


def confidence_score(
    row: dict[str, Any],
    *,
    branch_delta: float,
    branch_prior_weight: float,
    min_inliers: int,
) -> float:
    if row.get("te") is None or row.get("ae") is None:
        return -float("inf")
    inliers = max(0.0, _finite(row.get("inliers"), 0.0))
    if inliers < int(min_inliers):
        return -float("inf")
    return math.log1p(inliers) + float(branch_prior_weight) * float(branch_delta)


def select_query_rows(
    branch_rows: dict[str, list[dict[str, Any]]],
    *,
    baseline_branch: str,
    branch_deltas: dict[str, float],
    mode: str,
    branch_prior_weight: float,
    min_inliers: int,
    align_by: str,
) -> list[dict[str, Any]]:
    aligned = align_query_rows(branch_rows, align_by=align_by)
    selected = []
    for idx, candidates in enumerate(aligned):
        if mode == "oracle":
            best_branch = min(
                candidates,
                key=lambda branch: _finite(candidates[branch].get("te"), float("inf")),
            )
        else:
            scores = {
                branch: confidence_score(
                    row,
                    branch_delta=branch_deltas.get(branch, 0.0),
                    branch_prior_weight=branch_prior_weight,
                    min_inliers=min_inliers,
                )
                for branch, row in candidates.items()
            }
            best_branch = max(scores, key=scores.get)
            if not math.isfinite(scores[best_branch]) and baseline_branch in candidates:
                best_branch = baseline_branch
        row = candidates[best_branch]
        selected.append(
            {
                "query_index": idx,
                "image_name": row["image_name"],
                "selected_branch": best_branch,
                "te": row["te"],
                "ae": row["ae"],
                "inliers": row["inliers"],
            }
        )
    return selected


def summarize_selected_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return summarize_rows(
        [
            {
                "te": row["te"],
                "ae": row["ae"],
                "inliers": row["inliers"],
            }
            for row in rows
        ]
    )


def evaluate_query_selector(
    manifest_path: str | Path,
    *,
    stage: str = "dense",
    baseline_branch: str = "stdloc_baseline",
    mode: str = "calibrated_confidence",
    metric: str = "combined",
    te_penalty_per_cm: float = 0.002,
    branch_prior_weight: float = 5.0,
    min_inliers: int = 4,
    align_by: str = "auto",
    calibration_stride: int = 0,
    calibration_offset: int = 0,
    test_stride: int = 0,
    test_offset: int = 0,
) -> dict[str, Any]:
    manifest = load_branch_manifest(manifest_path)
    scene_outputs = []
    all_rows = []
    for scene, branches in manifest.items():
        branch_rows = {}
        calibration_rows = {}
        result_dirs = {}
        calibration_dirs = {}
        for branch, spec in branches.items():
            result_dir = spec.get("test_dir") or spec.get("eval_dir") or spec.get("result_dir") or spec.get("dir")
            if not result_dir:
                raise KeyError(f"missing result dir for {scene}/{branch}")
            calibration_dir = spec.get("calibration_dir") or spec.get("val_dir") or result_dir
            result_dirs[branch] = str(result_dir_from_path(result_dir))
            calibration_dirs[branch] = str(result_dir_from_path(calibration_dir))
            branch_rows[branch] = filter_rows(
                load_result_rows(result_dir, stage=stage),
                stride=test_stride,
                offset=test_offset,
            )
            calibration_rows[branch] = filter_rows(
                load_result_rows(calibration_dir, stage=stage),
                stride=calibration_stride,
                offset=calibration_offset,
            )
        branch_deltas = calibration_branch_deltas(
            calibration_rows,
            baseline_branch=baseline_branch,
            metric=metric,
            te_penalty_per_cm=te_penalty_per_cm,
        )
        selected_rows = select_query_rows(
            branch_rows,
            baseline_branch=baseline_branch,
            branch_deltas=branch_deltas,
            mode=mode,
            branch_prior_weight=branch_prior_weight,
            min_inliers=min_inliers,
            align_by=align_by,
        )
        summary = summarize_selected_rows(selected_rows)
        branch_counts: dict[str, int] = {}
        for row in selected_rows:
            row["scene"] = scene
            branch_counts[row["selected_branch"]] = branch_counts.get(row["selected_branch"], 0) + 1
        all_rows.extend(selected_rows)
        scene_outputs.append(
            {
                "scene": scene,
                "summary": summary,
                "branch_counts": branch_counts,
                "branch_deltas": branch_deltas,
                "result_dirs": result_dirs,
                "calibration_dirs": calibration_dirs,
                "queries": len(selected_rows),
            }
        )
    macro = {
        "mean_median_te": float(np.mean([row["summary"]["median_te"] for row in scene_outputs]))
        if scene_outputs
        else float("inf"),
        "mean_median_ae": float(np.mean([row["summary"]["median_ae"] for row in scene_outputs]))
        if scene_outputs
        else float("inf"),
        "macro_recall_5cm_5d": float(np.mean([row["summary"]["recall_5cm_5d"] for row in scene_outputs]))
        if scene_outputs
        else 0.0,
        "macro_recall_2cm_2d": float(np.mean([row["summary"]["recall_2cm_2d"] for row in scene_outputs]))
        if scene_outputs
        else 0.0,
        "queries": int(sum(row["queries"] for row in scene_outputs)),
    }
    return {
        "mode": mode,
        "stage": stage,
        "manifest": str(manifest_path),
        "baseline_branch": baseline_branch,
        "metric": metric,
        "te_penalty_per_cm": float(te_penalty_per_cm),
        "branch_prior_weight": float(branch_prior_weight),
        "min_inliers": int(min_inliers),
        "align_by": align_by,
        "calibration_stride": int(calibration_stride),
        "calibration_offset": int(calibration_offset),
        "test_stride": int(test_stride),
        "test_offset": int(test_offset),
        "scenes": scene_outputs,
        "macro": macro,
        "results": all_rows,
    }


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "median_te",
        "median_ae",
        "recall_5cm_5d",
        "recall_2cm_2d",
        "localized",
        "queries",
        "branch_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene in summary["scenes"]:
            metrics = scene["summary"]
            writer.writerow(
                {
                    "scene": scene["scene"],
                    "median_te": metrics["median_te"],
                    "median_ae": metrics["median_ae"],
                    "recall_5cm_5d": metrics["recall_5cm_5d"],
                    "recall_2cm_2d": metrics["recall_2cm_2d"],
                    "localized": metrics["localized"],
                    "queries": metrics["queries"],
                    "branch_counts": json.dumps(scene["branch_counts"], sort_keys=True),
                }
            )
        writer.writerow(
            {
                "scene": "MACRO_MEAN",
                "median_te": summary["macro"]["mean_median_te"],
                "median_ae": summary["macro"]["mean_median_ae"],
                "recall_5cm_5d": summary["macro"]["macro_recall_5cm_5d"],
                "recall_2cm_2d": summary["macro"]["macro_recall_2cm_2d"],
                "localized": "",
                "queries": summary["macro"]["queries"],
                "branch_counts": "",
            }
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select Cambridge poses per query from branch eval results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--baseline_branch", default="stdloc_baseline")
    parser.add_argument("--mode", choices=["calibrated_confidence", "oracle"], default="calibrated_confidence")
    parser.add_argument("--metric", choices=["combined", "r5", "median_te"], default="combined")
    parser.add_argument("--te_penalty_per_cm", type=float, default=0.002)
    parser.add_argument("--branch_prior_weight", type=float, default=5.0)
    parser.add_argument("--min_inliers", type=int, default=4)
    parser.add_argument("--align_by", choices=["auto", "index", "name"], default="auto")
    parser.add_argument("--calibration_stride", type=int, default=0)
    parser.add_argument("--calibration_offset", type=int, default=0)
    parser.add_argument("--test_stride", type=int, default=0)
    parser.add_argument("--test_offset", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_query_selector(
        args.manifest,
        stage=args.stage,
        baseline_branch=args.baseline_branch,
        mode=args.mode,
        metric=args.metric,
        te_penalty_per_cm=args.te_penalty_per_cm,
        branch_prior_weight=args.branch_prior_weight,
        min_inliers=args.min_inliers,
        align_by=args.align_by,
        calibration_stride=args.calibration_stride,
        calibration_offset=args.calibration_offset,
        test_stride=args.test_stride,
        test_offset=args.test_offset,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(summary, output_dir / "query_selected_table.csv")
    print(json.dumps(summary["macro"], indent=2))


if __name__ == "__main__":
    main()
