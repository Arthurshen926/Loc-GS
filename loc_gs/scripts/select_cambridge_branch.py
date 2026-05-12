#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from loc_gs.localization.pose_metrics import (
    POSE_RECALL_THRESHOLDS,
    pose_error_summary,
    recall_metric_key,
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _row_stage_value(row: dict[str, Any], stage: str, metric: str) -> float | None:
    lower = f"{stage}_{metric}"
    upper = f"{stage}_{metric.upper()}"
    value = _row_value(row, lower, upper)
    if value is not None:
        return value
    nested = row.get(stage)
    if isinstance(nested, dict):
        value = _row_value(nested, metric, metric.upper())
        if value is not None:
            return value
    if stage == "dense":
        dense = row.get("dense")
        if isinstance(dense, list) and dense:
            return _row_value(dense[-1], metric, metric.upper())
    return None


def _row_inliers(row: dict[str, Any], stage: str) -> int:
    value = _row_stage_value(row, stage, "inliers")
    if value is None:
        return 0
    return int(value)


def _normalise_image_key(value: str) -> str:
    return value.strip().replace("\\", "/")


def _row_key(row: dict[str, Any], index: int) -> str:
    for key in ("image_name", "query_name", "name", "image"):
        value = row.get(key)
        if value:
            return _normalise_image_key(str(value))
    return f"idx:{index:06d}"


def result_dir_from_path(path: str | Path) -> Path:
    path = Path(path)
    if path.name in {"summary.json", "results.json"}:
        return path.parent
    return path


def load_result_rows(path: str | Path, stage: str = "dense") -> list[dict[str, Any]]:
    result_dir = result_dir_from_path(path)
    results_path = result_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"missing results.json under {result_dir}")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("results", [])
    else:
        rows = data
    if not isinstance(rows, list):
        raise TypeError(f"results.json must contain a list or a dict with 'results': {results_path}")

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        te = _row_stage_value(row, stage, "te")
        ae = _row_stage_value(row, stage, "ae")
        out.append(
            {
                "image_name": _row_key(row, index),
                "source_index": index,
                "te": te,
                "ae": ae,
                "inliers": _row_inliers(row, stage),
                "localized": te is not None and ae is not None,
            }
        )
    return out


def load_calibration_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration id file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            values = data.get("ids") or data.get("image_names") or data.get("calibration_ids") or []
        else:
            values = data
        return {_normalise_image_key(str(value)) for value in values}
    return {
        _normalise_image_key(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    ids: set[str] | None = None,
    stride: int = 0,
    offset: int = 0,
) -> list[dict[str, Any]]:
    selected = rows
    if ids is not None:
        basenames = {Path(key).name for key in ids}
        selected = [
            row
            for row in selected
            if row["image_name"] in ids or Path(str(row["image_name"])).name in basenames
        ]
    if stride and stride > 1:
        selected = [row for row in selected if int(row["source_index"]) % int(stride) == int(offset)]
    return selected


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    te = [row["te"] for row in rows if row["te"] is not None and row["ae"] is not None]
    ae = [row["ae"] for row in rows if row["te"] is not None and row["ae"] is not None]
    inliers = [row["inliers"] for row in rows if row["te"] is not None and row["ae"] is not None]
    summary = pose_error_summary(te, ae, inliers)
    return {
        "queries": float(len(rows)),
        "localized": float(len(te)),
        **summary,
    }


def _branch_path(spec: Any, key: str) -> str:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"branch spec must be a string or dict, got {type(spec)!r}")
    for candidate in (key, "result_dir", "eval_dir", "dir", "summary", "summary_json"):
        value = spec.get(candidate)
        if value:
            path = Path(str(value))
            return str(path.parent if path.name == "summary.json" else path)
    raise KeyError(f"branch spec lacks {key}/result_dir/eval_dir: {spec}")


def load_branch_manifest(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = data.get("scenes", data)
    if not isinstance(scenes, dict):
        raise TypeError("branch manifest must be a scene mapping or contain a 'scenes' mapping")
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for scene, scene_spec in scenes.items():
        branches = scene_spec.get("branches", scene_spec) if isinstance(scene_spec, dict) else {}
        if not isinstance(branches, dict):
            raise TypeError(f"scene {scene} does not contain a branch mapping")
        out[str(scene)] = {}
        for branch, spec in branches.items():
            if isinstance(spec, str):
                out[str(scene)][str(branch)] = {"eval_dir": spec}
            elif isinstance(spec, dict):
                out[str(scene)][str(branch)] = dict(spec)
            else:
                raise TypeError(f"branch {scene}/{branch} has unsupported spec: {type(spec)!r}")
    return out


def branch_score(
    metrics: dict[str, float],
    baseline: dict[str, float] | None,
    metric: str,
    te_penalty_per_cm: float,
) -> float:
    if metric == "combined":
        score = float(metrics["recall_5cm_5d"])
        if baseline is not None:
            score -= float(te_penalty_per_cm) * max(0.0, float(metrics["median_te"]) - float(baseline["median_te"]))
        return score
    if metric == "median_te":
        return -float(metrics["median_te"])
    recall_key = recall_metric_key(metric)
    if recall_key in metrics:
        return float(metrics[recall_key])
    raise ValueError(f"unsupported target metric: {metric}")


def select_scene_branch(
    branch_metrics: dict[str, dict[str, float]],
    *,
    baseline_branch: str,
    metric: str = "combined",
    te_penalty_per_cm: float = 0.002,
    allow_r5_drop: float = 0.0,
    r5_tie: float = 0.01,
    candidate_min_r5: float = 0.0,
    max_median_te_increase_cm: float = math.inf,
    tie_prefer_branch: str = "",
    tie_max_te_increase_cm: float = 0.0,
    tie_max_r5_drop: float = 0.0,
    tie_min_r5: float = 0.0,
) -> tuple[str, dict[str, float]]:
    if not branch_metrics:
        raise ValueError("at least one candidate branch is required")
    baseline = branch_metrics.get(baseline_branch)
    candidates: list[tuple[str, float, dict[str, float]]] = []
    for name, metrics in branch_metrics.items():
        if baseline is not None and name != baseline_branch:
            if float(metrics["recall_5cm_5d"]) < float(candidate_min_r5):
                continue
            if float(metrics["recall_5cm_5d"]) < float(baseline["recall_5cm_5d"]) - float(allow_r5_drop):
                continue
            if float(metrics["median_te"]) > float(baseline["median_te"]) + float(max_median_te_increase_cm):
                continue
        score = branch_score(metrics, baseline, metric, te_penalty_per_cm)
        candidates.append((name, score, metrics))
    if not candidates:
        if baseline is None:
            raise ValueError("all candidates were filtered and no baseline branch is available")
        return baseline_branch, baseline
    candidates.sort(key=lambda item: (item[1], item[2]["recall_5cm_5d"], -item[2]["median_te"]), reverse=True)
    best_name, best_score, best_metrics = candidates[0]
    if baseline is not None and metric == "r5" and best_name != baseline_branch:
        if abs(float(best_metrics["recall_5cm_5d"]) - float(baseline["recall_5cm_5d"])) < float(r5_tie):
            if float(baseline["median_te"]) <= float(best_metrics["median_te"]):
                return baseline_branch, baseline
    if baseline is not None and best_name != baseline_branch:
        baseline_score = branch_score(baseline, baseline, metric, te_penalty_per_cm)
        if abs(best_score - baseline_score) < 1e-12 and float(baseline["median_te"]) <= float(best_metrics["median_te"]):
            return baseline_branch, baseline
    if (
        baseline is not None
        and best_name == baseline_branch
        and tie_prefer_branch
        and tie_prefer_branch in branch_metrics
    ):
        preferred = branch_metrics[tie_prefer_branch]
        preferred_r5 = float(preferred["recall_5cm_5d"])
        baseline_r5 = float(baseline["recall_5cm_5d"])
        preferred_te = float(preferred["median_te"])
        baseline_te = float(baseline["median_te"])
        if (
            preferred_r5 >= float(tie_min_r5)
            and preferred_r5 >= baseline_r5 - float(tie_max_r5_drop)
            and preferred_te <= baseline_te + float(tie_max_te_increase_cm)
        ):
            return tie_prefer_branch, preferred
    return best_name, best_metrics


def select_branches(
    manifest: dict[str, dict[str, dict[str, Any]]],
    *,
    stage: str,
    calibration_ids: set[str] | None,
    calibration_stride: int,
    calibration_offset: int,
    baseline_branch: str,
    metric: str,
    te_penalty_per_cm: float,
    allow_r5_drop: float,
    r5_tie: float,
    candidate_min_r5: float = 0.0,
    max_median_te_increase_cm: float = math.inf,
    tie_prefer_branch: str = "",
    tie_max_te_increase_cm: float = 0.0,
    tie_max_r5_drop: float = 0.0,
    tie_min_r5: float = 0.0,
) -> dict[str, Any]:
    scene_outputs: dict[str, Any] = {}
    selected: dict[str, str] = {}
    for scene, branches in manifest.items():
        metrics_by_branch: dict[str, dict[str, float]] = {}
        rows_by_branch: dict[str, int] = {}
        for branch, spec in branches.items():
            result_dir = _branch_path(spec, "calibration_dir")
            rows = load_result_rows(result_dir, stage=stage)
            rows = filter_rows(
                rows,
                ids=calibration_ids,
                stride=calibration_stride,
                offset=calibration_offset,
            )
            metrics_by_branch[branch] = summarize_rows(rows)
            rows_by_branch[branch] = len(rows)
        selected_branch, selected_metrics = select_scene_branch(
            metrics_by_branch,
            baseline_branch=baseline_branch,
            metric=metric,
            te_penalty_per_cm=te_penalty_per_cm,
            allow_r5_drop=allow_r5_drop,
            r5_tie=r5_tie,
            candidate_min_r5=candidate_min_r5,
            max_median_te_increase_cm=max_median_te_increase_cm,
            tie_prefer_branch=tie_prefer_branch,
            tie_max_te_increase_cm=tie_max_te_increase_cm,
            tie_max_r5_drop=tie_max_r5_drop,
            tie_min_r5=tie_min_r5,
        )
        selected[scene] = selected_branch
        selected_spec = branches[selected_branch]
        scene_outputs[scene] = {
            "selected_branch": selected_branch,
            "selected_dir": _branch_path(selected_spec, "test_dir"),
            "selected_metrics": selected_metrics,
            "branches": metrics_by_branch,
            "calibration_rows": rows_by_branch,
        }
    return {
        "selected_branch": selected,
        "scenes": scene_outputs,
        "selection": {
            "stage": stage,
            "metric": metric,
            "baseline_branch": baseline_branch,
            "te_penalty_per_cm": float(te_penalty_per_cm),
            "allow_r5_drop": float(allow_r5_drop),
            "r5_tie": float(r5_tie),
            "candidate_min_r5": float(candidate_min_r5),
            "max_median_te_increase_cm": float(max_median_te_increase_cm),
            "tie_prefer_branch": str(tie_prefer_branch),
            "tie_max_te_increase_cm": float(tie_max_te_increase_cm),
            "tie_max_r5_drop": float(tie_max_r5_drop),
            "tie_min_r5": float(tie_min_r5),
            "calibration_stride": int(calibration_stride),
            "calibration_offset": int(calibration_offset),
            "uses_calibration_ids": calibration_ids is not None,
        },
    }


def write_selection_csv(selection: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recall_fields = [key for key, _te_thr, _ae_thr in POSE_RECALL_THRESHOLDS]
    fieldnames = [
        "scene",
        "selected_branch",
        "median_te",
        "median_ae",
        *recall_fields,
        "localized",
        "queries",
        "selected_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene, data in selection["scenes"].items():
            metrics = data["selected_metrics"]
            writer.writerow(
                {
                    "scene": scene,
                    "selected_branch": data["selected_branch"],
                    "median_te": metrics["median_te"],
                    "median_ae": metrics["median_ae"],
                    **{key: metrics.get(key, 0.0) for key in recall_fields},
                    "localized": int(metrics["localized"]),
                    "queries": int(metrics["queries"]),
                    "selected_dir": data["selected_dir"],
                }
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select Cambridge localization branches from a calibration-val split.")
    parser.add_argument("--manifest", required=True, help="JSON mapping scene -> branch -> result directories")
    parser.add_argument("--output", required=True, help="Path to selected_branch.json")
    parser.add_argument("--stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--calibration_ids", default="", help="Optional txt/json list of calibration-val image names")
    parser.add_argument("--calibration_stride", type=int, default=0, help="Use every Nth result row as calibration-val")
    parser.add_argument("--calibration_offset", type=int, default=0)
    parser.add_argument("--baseline_branch", default="stdloc_baseline")
    parser.add_argument(
        "--metric",
        choices=["combined", "median_te", "r5", "r2", "r10", "r25", "r50", "r1m", "r2m", "r5m"],
        default="combined",
    )
    parser.add_argument("--te_penalty_per_cm", type=float, default=0.002)
    parser.add_argument("--allow_r5_drop", type=float, default=0.0)
    parser.add_argument("--r5_tie", type=float, default=0.01)
    parser.add_argument(
        "--candidate_min_r5",
        type=float,
        default=0.0,
        help="Reject non-baseline branches whose calibration R@5cm falls below this floor.",
    )
    parser.add_argument(
        "--max_median_te_increase_cm",
        type=float,
        default=math.inf,
        help="Reject non-baseline branches whose calibration median error exceeds baseline by this margin.",
    )
    parser.add_argument(
        "--tie_prefer_branch",
        default="",
        help="Prefer this branch over the baseline when calibration metrics are within the tie tolerances.",
    )
    parser.add_argument("--tie_max_te_increase_cm", type=float, default=0.0)
    parser.add_argument("--tie_max_r5_drop", type=float, default=0.0)
    parser.add_argument("--tie_min_r5", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    manifest = load_branch_manifest(args.manifest)
    calibration_ids = load_calibration_ids(args.calibration_ids) if args.calibration_ids else None
    selection = select_branches(
        manifest,
        stage=args.stage,
        calibration_ids=calibration_ids,
        calibration_stride=args.calibration_stride,
        calibration_offset=args.calibration_offset,
        baseline_branch=args.baseline_branch,
        metric=args.metric,
        te_penalty_per_cm=args.te_penalty_per_cm,
        allow_r5_drop=args.allow_r5_drop,
        r5_tie=args.r5_tie,
        candidate_min_r5=args.candidate_min_r5,
        max_median_te_increase_cm=args.max_median_te_increase_cm,
        tie_prefer_branch=args.tie_prefer_branch,
        tie_max_te_increase_cm=args.tie_max_te_increase_cm,
        tie_max_r5_drop=args.tie_max_r5_drop,
        tie_min_r5=args.tie_min_r5,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_selection_csv(selection, output.with_suffix(".csv"))
    print(json.dumps(selection["selected_branch"], indent=2))


if __name__ == "__main__":
    main()
