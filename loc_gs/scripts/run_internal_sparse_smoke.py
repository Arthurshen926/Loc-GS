#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Sequence

from loc_gs.core.camera import load_camera_records
from loc_gs.core.metrics import pose_error_cm_deg
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.landmarks import CacheLandmarkResolver, load_gaussian_landmark_map
from loc_gs.sparse.pipeline import SparseLocalizationConfig, run_sparse_localization
from loc_gs.sparse.real_inputs import CachedSparseInputConfig, sparse_input_from_cached_batch
from loc_gs.training.sparse_candidate_scorer import candidate_solver_score_rows, load_candidate_scorer


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_smoke_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    candidate_scorer: str | Path | None = None,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal sparse smoke manifest")
    return {
        "schema_version": "internal_sparse_smoke_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_only",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(candidate_artifact),
        "point_cloud": str(point_cloud),
        "cameras_json": str(cameras_json),
        "candidate_scorer": None if candidate_scorer is None else str(candidate_scorer),
        "hyperparameters": hyperparameters,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small internal sparse cached-candidate PnP smoke.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--missing_principal_point", choices=["half_extent", "pixel_center"], default="pixel_center")
    parser.add_argument("--max_queries", type=int, default=5)
    parser.add_argument("--max_keypoints", type=int, default=512)
    parser.add_argument("--score_mode", choices=["native", "teacher_oracle"], default="native")
    parser.add_argument("--candidate_scorer", type=Path, default=None)
    parser.add_argument("--rerank_prefix_fraction", type=float, default=1.0)
    parser.add_argument("--solver_weight", type=float, default=1.0)
    parser.add_argument("--native_weight", type=float, default=1.0)
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=10000)
    parser.add_argument("--pnp_method", choices=["epnp", "iterative"], default="epnp")
    parser.add_argument("--refine_with_inliers", action="store_true")
    parser.add_argument("--second_pnp_enabled", action="store_true")
    parser.add_argument("--second_pnp_method", choices=["epnp", "iterative"], default="iterative")
    parser.add_argument("--lgcv_reprojection_error_px", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    reject_test_split(str(args.split_name), purpose="internal sparse smoke")
    if (args.image_width is None) != (args.image_height is None):
        raise ValueError("--image_width and --image_height must be provided together")
    command = [sys.executable, "-m", "loc_gs.scripts.run_internal_sparse_smoke", *(argv or sys.argv[1:])]
    artifact = load_listwise_candidate_artifact(args.candidate_artifact)
    landmark_map = load_gaussian_landmark_map(args.point_cloud)
    resolver = CacheLandmarkResolver.from_pair_cache(args.candidate_artifact, landmark_map)
    scorer = load_candidate_scorer(args.candidate_scorer) if args.candidate_scorer is not None else None
    if args.image_width is None:
        cameras = load_camera_records(args.cameras_json)
        pose_metric_frame = "camera_json_c2w"
    else:
        cameras = load_camera_records(
            args.cameras_json,
            target_width=int(args.image_width),
            target_height=int(args.image_height),
            missing_principal_point=str(args.missing_principal_point),
        )
        pose_metric_frame = f"camera_json_c2w_resized_{args.missing_principal_point}"

    input_cfg = CachedSparseInputConfig(score_mode=args.score_mode, max_keypoints=int(args.max_keypoints))
    loc_cfg = SparseLocalizationConfig(
        rerank_prefix_fraction=float(args.rerank_prefix_fraction),
        solver_weight=float(args.solver_weight),
        native_weight=float(args.native_weight),
        reprojection_error_px=float(args.reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
        pnp_method=str(args.pnp_method),
        refine_with_inliers=bool(args.refine_with_inliers),
        second_pnp_enabled=bool(args.second_pnp_enabled),
        second_pnp_method=str(args.second_pnp_method),
        lgcv_reprojection_error_px=float(args.lgcv_reprojection_error_px),
    )
    rows: list[dict[str, object]] = []
    te_cm_values: list[float] = []
    re_deg_values: list[float] = []
    stage_counts: list[float] = []
    lgcv_keep_counts: list[float] = []
    for batch in artifact.batches[: max(0, int(args.max_queries))]:
        camera = cameras.get(batch.query_id)
        if camera is None:
            rows.append({"query_id": batch.query_id, "success": False, "reason": "missing_camera"})
            continue
        batch_input_cfg = input_cfg
        if scorer is not None:
            batch_input_cfg = CachedSparseInputConfig(
                score_mode=args.score_mode,
                max_keypoints=int(args.max_keypoints),
                solver_score_rows=candidate_solver_score_rows(batch, scorer),
            )
        data = sparse_input_from_cached_batch(batch, resolver, intrinsics=camera.intrinsics, cfg=batch_input_cfg)
        result = run_sparse_localization(data, loc_cfg)
        te_cm = None
        re_deg = None
        if result.success and result.pose_w2c is not None and camera.pose_w2c is not None:
            te_cm, re_deg = pose_error_cm_deg(result.pose_w2c, camera.pose_w2c)
            te_cm_values.append(float(te_cm))
            re_deg_values.append(float(re_deg))
        stage_counts.append(float(result.pnp_stage_count))
        if result.lgcv_keep_count is not None:
            lgcv_keep_counts.append(float(result.lgcv_keep_count))
        rows.append(
            {
                "query_id": batch.query_id,
                "success": bool(result.success),
                "inlier_count": int(result.inlier_count),
                "initial_inlier_count": int(result.initial_inlier_count),
                "lgcv_keep_count": result.lgcv_keep_count,
                "pnp_stage_count": int(result.pnp_stage_count),
                "selected_count": int(len(result.selected_landmark_ids)),
                "te_cm": te_cm,
                "re_deg": re_deg,
                "availability_summary": result.availability_summary,
            }
        )

    success_rows = [row for row in rows if bool(row.get("success"))]
    inliers = [int(row.get("inlier_count", 0)) for row in rows]
    summary = {
        "schema_version": "internal_sparse_smoke_metrics_v1",
        "scene": str(args.scene),
        "split_name": str(args.split_name),
        "query_count": int(len(rows)),
        "success_count": int(len(success_rows)),
        "success_rate": float(len(success_rows) / max(1, len(rows))),
        "mean_inliers": float(mean(inliers)) if inliers else 0.0,
        "median_te_cm": _median_or_none(te_cm_values),
        "median_re_deg": _median_or_none(re_deg_values),
        "pose_metric_status": "computed_unverified" if te_cm_values else "missing_gt_pose",
        "pose_metric_frame": pose_metric_frame,
        "score_mode": str(args.score_mode),
        "candidate_scorer_enabled": bool(scorer is not None),
        "pnp_stage_count_median": _median_or_none(stage_counts),
        "lgcv_keep_count_median": _median_or_none(lgcv_keep_counts),
        "candidate_artifact": artifact.summarize_candidate_availability(),
    }
    hyperparameters = {
        "image_width": args.image_width,
        "image_height": args.image_height,
        "missing_principal_point": str(args.missing_principal_point),
        "max_queries": int(args.max_queries),
        "max_keypoints": int(args.max_keypoints),
        "score_mode": str(args.score_mode),
        "candidate_scorer": None if args.candidate_scorer is None else str(args.candidate_scorer),
        "rerank_prefix_fraction": float(args.rerank_prefix_fraction),
        "solver_weight": float(args.solver_weight),
        "native_weight": float(args.native_weight),
        "reprojection_error_px": float(args.reprojection_error_px),
        "pnp_iterations": int(args.pnp_iterations),
        "pnp_method": str(args.pnp_method),
        "refine_with_inliers": bool(args.refine_with_inliers),
        "second_pnp_enabled": bool(args.second_pnp_enabled),
        "second_pnp_method": str(args.second_pnp_method),
        "lgcv_reprojection_error_px": float(args.lgcv_reprojection_error_px),
    }
    manifest = build_smoke_manifest(
        scene=str(args.scene),
        split_name=str(args.split_name),
        command=command,
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        candidate_scorer=args.candidate_scorer,
        hyperparameters=hyperparameters,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
