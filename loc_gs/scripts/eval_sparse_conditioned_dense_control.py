#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))

from loc_gs.scripts.visualize_stdloc_hard_matches import (
    _build_context,
    _capture_dense,
    _capture_sparse,
    _resolve_slcdp_effective_options,
    pose_error_cm_deg,
)


CAMBRIDGE_SCENES = ("GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch")
RECALL_THRESHOLDS = {
    "R50cm5deg": (50.0, 5.0),
    "R15cm5deg": (15.0, 5.0),
    "R10cm5deg": (10.0, 5.0),
    "R5cm5deg": (5.0, 5.0),
    "R2cm2deg": (2.0, 2.0),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True).strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=_repo_root(), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _preload_render_backend() -> bool:
    """Load gsplat's CUDA extension before large scene objects inflate process memory."""

    try:
        import gsplat.cuda._backend  # noqa: F401
    except Exception:
        return False
    return True


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _cvar_tail(values: Sequence[float], fraction: float = 0.10) -> float | None:
    if not values:
        return None
    ordered = sorted((float(v) for v in values), reverse=True)
    count = max(1, int(np.ceil(len(ordered) * float(fraction))))
    return float(sum(ordered[:count]) / count)


def summarize_pose_rows(rows: Sequence[Mapping[str, Any]], *, te_key: str, re_key: str) -> dict[str, Any]:
    te_values = [float(row[te_key]) for row in rows if row.get(te_key) is not None and np.isfinite(float(row[te_key]))]
    re_values = [float(row[re_key]) for row in rows if row.get(re_key) is not None and np.isfinite(float(row[re_key]))]
    summary: dict[str, Any] = {
        "query_count": int(len(rows)),
        "valid_count": int(min(len(te_values), len(re_values))),
        "median_te_cm": float(np.median(te_values)) if te_values else None,
        "median_re_deg": float(np.median(re_values)) if re_values else None,
        "p90_te_cm": _percentile(te_values, 90.0),
        "p95_te_cm": _percentile(te_values, 95.0),
        "cvar10_te_cm": _cvar_tail(te_values, 0.10),
        "severe_rate_1m": float(sum(v > 100.0 for v in te_values) / len(te_values)) if te_values else None,
        "catastrophic_rate_5m": float(sum(v > 500.0 for v in te_values) / len(te_values)) if te_values else None,
    }
    for name, (te_thr, re_thr) in RECALL_THRESHOLDS.items():
        hits = [
            float(row[te_key]) <= te_thr and float(row[re_key]) <= re_thr
            for row in rows
            if row.get(te_key) is not None and row.get(re_key) is not None
        ]
        summary[name] = float(sum(hits) / len(hits)) if hits else None
    return summary


def summarize_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(row["sparse_conditioned_dense_te_cm"]) - float(row["base_dense_te_cm"])
        for row in rows
        if row.get("sparse_conditioned_dense_te_cm") is not None and row.get("base_dense_te_cm") is not None
    ]
    return {
        "query_count": int(len(rows)),
        "median_delta_te_cm": float(np.median(deltas)) if deltas else None,
        "mean_delta_te_cm": float(np.mean(deltas)) if deltas else None,
        "p90_delta_te_cm": _percentile(deltas, 90.0),
        "regression_20cm_count": int(sum(v >= 20.0 for v in deltas)),
        "regression_50cm_count": int(sum(v >= 50.0 for v in deltas)),
        "improvement_20cm_count": int(sum(v <= -20.0 for v in deltas)),
        "worse_gt_0_1cm_count": int(sum(v > 0.1 for v in deltas)),
        "worse_gt_0_5cm_count": int(sum(v > 0.5 for v in deltas)),
        "improved_gt_0_5cm_count": int(sum(v < -0.5 for v in deltas)),
        "non_base_render_count": int(sum(str(row.get("sparse_conditioned_label")) != "base" for row in rows)),
        "low_confidence_gated_base_count": int(sum(bool(row.get("low_confidence_gated_base")) for row in rows)),
        "guided_pose_count": int(sum(str(row.get("sparse_conditioned_label", "")).startswith("gated_ray_") for row in rows)),
        "base_dense_reused_count": int(sum(bool(row.get("base_dense_reused")) for row in rows)),
        "base_dense_high_confidence_reuse_count": int(
            sum(str(row.get("base_reuse_reason", "")) == "base_dense_high_confidence" for row in rows)
        ),
        "base_preflight_safe_reuse_count": int(
            sum(str(row.get("base_reuse_reason", "")) == "base_preflight_safe" for row in rows)
        ),
        "base_dense_worsened_count": int(
            sum(
                float(row["base_dense_te_cm"]) - float(row["sparse_te_cm"]) >= 5.0
                for row in rows
                if row.get("base_dense_te_cm") is not None and row.get("sparse_te_cm") is not None
            )
        ),
        "sparse_conditioned_dense_worsened_count": int(
            sum(
                float(row["sparse_conditioned_dense_te_cm"]) - float(row["sparse_te_cm"]) >= 5.0
                for row in rows
                if row.get("sparse_conditioned_dense_te_cm") is not None and row.get("sparse_te_cm") is not None
            )
        ),
    }


def _fixed_options(render_control: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    from loc_gs.scripts.visualize_stdloc_hard_matches import build_argparser as build_visualizer_argparser

    parser = build_visualizer_argparser()
    args = parser.parse_args(["--slcdp_render_control", render_control])
    return _resolve_slcdp_effective_options(args)


def _scene_output_dir(output_dir: Path, scene: str) -> Path:
    return output_dir / scene


def _resume_rows(scene_dir: Path, cameras: Sequence[Any], *, resume_partial: bool) -> list[dict[str, Any]]:
    if not bool(resume_partial):
        return []
    partial_path = scene_dir / "results.partial.json"
    if not partial_path.exists():
        return []
    rows = json.loads(partial_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Partial results are not a row list: {partial_path}")
    if len(rows) > len(cameras):
        raise ValueError(f"Partial row count exceeds camera count: {len(rows)} > {len(cameras)}")
    for index, row in enumerate(rows):
        expected_name = str(getattr(cameras[index], "image_name"))
        actual_name = str(row.get("image_name"))
        if actual_name != expected_name:
            raise ValueError(
                f"Partial row {index} image_name={actual_name!r} does not match camera prefix {expected_name!r}"
            )
    return [dict(row) for row in rows]


def _resume_partial_rows(scene_dir: Path, *, resume_partial: bool) -> list[dict[str, Any]]:
    if not bool(resume_partial):
        return []
    partial_path = scene_dir / "results.partial.json"
    if not partial_path.exists():
        return []
    rows = json.loads(partial_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Partial results are not a row list: {partial_path}")
    return [dict(row) for row in rows]


def _selected_camera_count(camera_iterable: Any, *, query_stride: int, max_queries: int) -> int | None:
    try:
        total = int(len(camera_iterable))
    except TypeError:
        return int(max_queries) if int(max_queries) > 0 else None
    stride = max(1, int(query_stride))
    selected = (total + stride - 1) // stride
    if int(max_queries) > 0:
        selected = min(selected, int(max_queries))
    return int(selected)


def evaluate_scene(
    *,
    candidate_root: Path,
    output_dir: Path,
    scene: str,
    eval_split: str,
    max_queries: int,
    query_stride: int,
    progress_interval: int,
    resume_partial: bool = False,
) -> dict[str, Any]:
    _preload_render_backend()
    run_dir = candidate_root / scene
    ctx = _build_context(run_dir, split_override=eval_split)
    base_options = _fixed_options("none")
    candidate_options = _fixed_options("sparse_conditioned")
    cameras = ctx["cameras"]
    total_selected = _selected_camera_count(
        cameras,
        query_stride=int(query_stride),
        max_queries=int(max_queries),
    )
    scene_dir = _scene_output_dir(output_dir, scene)
    rows: list[dict[str, Any]] = _resume_partial_rows(scene_dir, resume_partial=bool(resume_partial))
    resume_start_index = int(len(rows))
    start = time.time()
    selected_index = 0
    for raw_index, camera in enumerate(cameras):
        if int(query_stride) > 1 and raw_index % int(query_stride) != 0:
            continue
        if int(max_queries) > 0 and selected_index >= int(max_queries):
            break
        if selected_index < resume_start_index:
            expected_name = str(rows[selected_index].get("image_name"))
            actual_name = str(getattr(camera, "image_name"))
            if actual_name != expected_name:
                raise ValueError(
                    f"Partial row {selected_index} image_name={expected_name!r} "
                    f"does not match camera prefix {actual_name!r}"
                )
            selected_index += 1
            continue
        index = selected_index
        query_image = camera.original_image.to("cuda")
        gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        with torch.no_grad():
            sparse = _capture_sparse(ctx, query_image, camera.FoVx, camera.FoVy)
            base_dense = _capture_dense(
                ctx,
                sparse["coarse_map"],
                sparse["fine_map"],
                sparse["pose_w2c"],
                camera.FoVx,
                camera.FoVy,
                sparse_capture=sparse,
                **base_options,
            )
            controlled_dense = _capture_dense(
                ctx,
                sparse["coarse_map"],
                sparse["fine_map"],
                sparse["pose_w2c"],
                camera.FoVx,
                camera.FoVy,
                sparse_capture=sparse,
                slcdp_base_dense_capture=base_dense,
                **candidate_options,
            )
        sparse_te, sparse_re = pose_error_cm_deg(sparse["pose_w2c"], gt_w2c)
        base_te, base_re = pose_error_cm_deg(base_dense["pose_w2c"], gt_w2c)
        controlled_te, controlled_re = pose_error_cm_deg(controlled_dense["pose_w2c"], gt_w2c)
        selection = (controlled_dense.get("slcdp_repair_search") or {}).get("selection") or {}
        render_control = controlled_dense.get("slcdp_render_control") or {}
        dense_quality = controlled_dense.get("dense_pose_quality") or {}
        rows.append(
            {
                "scene": scene,
                "split": eval_split,
                "query_index": int(index),
                "image_name": str(camera.image_name),
                "sparse_te_cm": sparse_te,
                "sparse_re_deg": sparse_re,
                "sparse_inlier_count": int(np.asarray(sparse.get("inliers", [])).reshape(-1).shape[0]),
                "base_dense_te_cm": base_te,
                "base_dense_re_deg": base_re,
                "base_dense_inlier_count": int(np.asarray(base_dense.get("inliers", [])).reshape(-1).shape[0]),
                "sparse_conditioned_dense_te_cm": controlled_te,
                "sparse_conditioned_dense_re_deg": controlled_re,
                "sparse_conditioned_dense_inlier_count": int(np.asarray(controlled_dense.get("inliers", [])).reshape(-1).shape[0]),
                "sparse_conditioned_label": render_control.get("candidate_label"),
                "sparse_conditioned_decision": selection.get("decision"),
                "sparse_conditioned_best_label": selection.get("best_label"),
                "sparse_conditioned_score_gain": selection.get("score_gain"),
                "low_confidence_gated_base": bool(selection.get("low_confidence_gated_base", False)),
                "transition_decision": (controlled_dense.get("slcdp_transition_control") or {}).get("decision"),
                "base_dense_reused": bool(render_control.get("base_dense_reused", False)),
                "base_reuse_reason": render_control.get("base_reuse_reason"),
                "dense_pose_match_count": dense_quality.get("match_count"),
                "dense_pose_solver_inlier_count": dense_quality.get("solver_inlier_count"),
                "dense_pose_solver_inlier_ratio": dense_quality.get("solver_inlier_ratio"),
                "dense_pose_median_reprojection_error_px": dense_quality.get("median_reprojection_error_px"),
                "dense_pose_p90_reprojection_error_px": dense_quality.get("p90_reprojection_error_px"),
            }
        )
        if (index + 1) % max(1, int(progress_interval)) == 0 or index == 0:
            _write_json(scene_dir / "results.partial.json", rows)
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "split": eval_split,
                        "processed": index + 1,
                        "total": total_selected,
                        "elapsed_sec": round(float(time.time() - start), 3),
                    }
                ),
                flush=True,
            )
        selected_index += 1
    summary = {
        "scene": scene,
        "split": eval_split,
        "elapsed_sec": float(time.time() - start),
        "base_dense": summarize_pose_rows(rows, te_key="base_dense_te_cm", re_key="base_dense_re_deg"),
        "sparse_conditioned_dense": summarize_pose_rows(
            rows,
            te_key="sparse_conditioned_dense_te_cm",
            re_key="sparse_conditioned_dense_re_deg",
        ),
        "sparse": summarize_pose_rows(rows, te_key="sparse_te_cm", re_key="sparse_re_deg"),
        "comparison": summarize_comparison(rows),
    }
    _write_json(scene_dir / "results.json", rows)
    _write_json(scene_dir / "summary.json", summary)
    _write_json(
        scene_dir / "manifest.json",
        {
            "schema": "loc_gs_sparse_conditioned_dense_control_eval_v1",
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "scene": scene,
            "split": eval_split,
            "candidate_root": str(candidate_root),
            "run_dir": str(run_dir),
            "map_path": ctx.get("map_path"),
            "data_root": ctx.get("scene_root"),
            "render_control": "sparse_conditioned",
            "base_render_control": "none",
            "selection_policy": candidate_options,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resume_partial": bool(resume_partial),
            "resume_start_index": int(resume_start_index),
            "diagnostic_only": True,
        },
    )
    (scene_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    (scene_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    _write_json(scene_dir / "split_audit.json", {"scene": scene, "split": eval_split, "diagnostic_only": True})
    return summary


def _macro_summary(scene_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"scene_count": int(len(scene_summaries))}
    for method in ("base_dense", "sparse_conditioned_dense"):
        method_rows = [summary[method] for summary in scene_summaries]
        out[method] = {
            key: float(np.mean([float(row[key]) for row in method_rows if row.get(key) is not None]))
            for key in ("median_te_cm", "median_re_deg", "R50cm5deg", "R15cm5deg", "R10cm5deg", "R5cm5deg", "R2cm2deg", "p90_te_cm", "p95_te_cm", "cvar10_te_cm")
        }
    comparisons = [summary["comparison"] for summary in scene_summaries]
    out["comparison"] = {
        "mean_median_delta_te_cm": float(np.mean([float(row["median_delta_te_cm"]) for row in comparisons if row.get("median_delta_te_cm") is not None])),
        "total_regression_20cm_count": int(sum(int(row.get("regression_20cm_count", 0)) for row in comparisons)),
        "total_regression_50cm_count": int(sum(int(row.get("regression_50cm_count", 0)) for row in comparisons)),
        "total_improvement_20cm_count": int(sum(int(row.get("improvement_20cm_count", 0)) for row in comparisons)),
        "total_non_base_render_count": int(sum(int(row.get("non_base_render_count", 0)) for row in comparisons)),
        "total_low_confidence_gated_base_count": int(sum(int(row.get("low_confidence_gated_base_count", 0)) for row in comparisons)),
        "total_guided_pose_count": int(sum(int(row.get("guided_pose_count", 0)) for row in comparisons)),
        "total_base_dense_reused_count": int(sum(int(row.get("base_dense_reused_count", 0)) for row in comparisons)),
        "total_base_dense_high_confidence_reuse_count": int(
            sum(int(row.get("base_dense_high_confidence_reuse_count", 0)) for row in comparisons)
        ),
        "total_base_preflight_safe_reuse_count": int(
            sum(int(row.get("base_preflight_safe_reuse_count", 0)) for row in comparisons)
        ),
        "base_dense_worsened_count": int(sum(int(row.get("base_dense_worsened_count", 0)) for row in comparisons)),
        "sparse_conditioned_dense_worsened_count": int(sum(int(row.get("sparse_conditioned_dense_worsened_count", 0)) for row in comparisons)),
    }
    return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fixed sparse-conditioned dense render control on Cambridge splits.")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--resume_partial", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    ns = build_argparser().parse_args() if args is None else args
    scenes = tuple(ns.scene) if ns.scene else CAMBRIDGE_SCENES
    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        evaluate_scene(
            candidate_root=Path(ns.candidate_root),
            output_dir=output_dir,
            scene=scene,
            eval_split=str(ns.eval_split),
            max_queries=int(ns.max_queries),
            query_stride=int(ns.query_stride),
            progress_interval=int(ns.progress_interval),
            resume_partial=bool(ns.resume_partial),
        )
        for scene in scenes
    ]
    macro = _macro_summary(summaries)
    _write_json(output_dir / "summary.json", {"scenes": summaries, "macro": macro})
    print(json.dumps({"output_dir": str(output_dir), "macro": macro}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
