#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from loc_gs.localization.pose_metrics import POSE_RECALL_THRESHOLDS
from loc_gs.scripts.select_cambridge_branch import (
    _branch_path,
    load_branch_manifest,
    load_result_rows,
    result_dir_from_path,
    summarize_rows,
)


def load_selected_branch(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "selected_branch" in data:
        return {str(scene): str(branch) for scene, branch in data["selected_branch"].items()}
    if isinstance(data, dict) and all(isinstance(value, str) for value in data.values()):
        return {str(scene): str(branch) for scene, branch in data.items()}
    raise TypeError(f"unsupported selected branch file: {path}")


def _selected_dir_from_selection(path: str | Path, scene: str) -> str | None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    scenes = data.get("scenes")
    if not isinstance(scenes, dict):
        return None
    scene_data = scenes.get(scene)
    if not isinstance(scene_data, dict):
        return None
    value = scene_data.get("selected_dir")
    return str(value) if value else None


def evaluate_selected_branches(
    selected_path: str | Path,
    manifest_path: str | Path | None,
    *,
    stage: str = "dense",
) -> dict[str, Any]:
    selected = load_selected_branch(selected_path)
    manifest = load_branch_manifest(manifest_path) if manifest_path else {}
    rows: list[dict[str, Any]] = []
    for scene, branch in selected.items():
        result_dir = _selected_dir_from_selection(selected_path, scene)
        branch_spec: dict[str, Any] | None = None
        if manifest:
            try:
                branch_spec = manifest[scene][branch]
            except KeyError as exc:
                raise KeyError(f"selected branch {scene}/{branch} not found in manifest") from exc
            result_dir = _branch_path(branch_spec, "test_dir")
        if not result_dir:
            raise KeyError(f"no result directory available for selected branch {scene}/{branch}")
        result_rows = load_result_rows(result_dir, stage=stage)
        metrics = summarize_rows(result_rows)
        rows.append(
            {
                "scene": scene,
                "selected_branch": branch,
                "result_dir": str(result_dir_from_path(result_dir)),
                **metrics,
            }
        )
    macro = {
        "mean_median_te": float(np.mean([row["median_te"] for row in rows])) if rows else float("inf"),
        "mean_median_ae": float(np.mean([row["median_ae"] for row in rows])) if rows else float("inf"),
        "queries": int(sum(row["queries"] for row in rows)),
        "localized": int(sum(row["localized"] for row in rows)),
    }
    for key, _te_thr, _ae_thr in POSE_RECALL_THRESHOLDS:
        macro[f"macro_{key}"] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    return {
        "stage": stage,
        "selected_branch_file": str(selected_path),
        "manifest": "" if manifest_path is None else str(manifest_path),
        "rows": rows,
        "macro": macro,
    }


def write_table_csv(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recall_fields = [key for key, _te_thr, _ae_thr in POSE_RECALL_THRESHOLDS]
    fieldnames = [
        "scene",
        "selected_branch",
        "median_te",
        "median_ae",
        *recall_fields,
        "avg_inliers",
        "localized",
        "queries",
        "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(row)
        writer.writerow(
            {
                "scene": "MACRO_MEAN",
                "selected_branch": "",
                "median_te": summary["macro"]["mean_median_te"],
                "median_ae": summary["macro"]["mean_median_ae"],
                **{key: summary["macro"][f"macro_{key}"] for key in recall_fields},
                "avg_inliers": "",
                "localized": summary["macro"]["localized"],
                "queries": summary["macro"]["queries"],
                "result_dir": "",
            }
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Cambridge test results using selected_branch.json.")
    parser.add_argument("--selected_branch", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", choices=["sparse", "dense"], default="dense")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_selected_branches(
        args.selected_branch,
        args.manifest or None,
        stage=args.stage,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table_csv(summary, output_dir / "cambridge_branch_selected_table.csv")
    print(json.dumps(summary["macro"], indent=2))


if __name__ == "__main__":
    main()
