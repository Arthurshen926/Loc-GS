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
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.landmarks import CacheLandmarkResolver, load_gaussian_landmark_map
from loc_gs.sparse.pipeline import SparseLocalizationConfig, run_sparse_localization
from loc_gs.sparse.real_inputs import CachedSparseInputConfig, sparse_input_from_cached_batch


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
    parser.add_argument("--max_queries", type=int, default=5)
    parser.add_argument("--max_keypoints", type=int, default=512)
    parser.add_argument("--score_mode", choices=["native", "teacher_oracle"], default="native")
    parser.add_argument("--rerank_prefix_fraction", type=float, default=1.0)
    parser.add_argument("--solver_weight", type=float, default=1.0)
    parser.add_argument("--native_weight", type=float, default=1.0)
    parser.add_argument("--reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    reject_test_split(str(args.split_name), purpose="internal sparse smoke")
    command = [sys.executable, "-m", "loc_gs.scripts.run_internal_sparse_smoke", *(argv or sys.argv[1:])]
    artifact = load_listwise_candidate_artifact(args.candidate_artifact)
    landmark_map = load_gaussian_landmark_map(args.point_cloud)
    resolver = CacheLandmarkResolver.from_pair_cache(args.candidate_artifact, landmark_map)
    cameras = load_camera_records(args.cameras_json)

    input_cfg = CachedSparseInputConfig(score_mode=args.score_mode, max_keypoints=int(args.max_keypoints))
    loc_cfg = SparseLocalizationConfig(
        rerank_prefix_fraction=float(args.rerank_prefix_fraction),
        solver_weight=float(args.solver_weight),
        native_weight=float(args.native_weight),
        reprojection_error_px=float(args.reprojection_error_px),
        pnp_iterations=int(args.pnp_iterations),
    )
    rows: list[dict[str, object]] = []
    for batch in artifact.batches[: max(0, int(args.max_queries))]:
        camera = cameras.get(batch.query_id)
        if camera is None:
            rows.append({"query_id": batch.query_id, "success": False, "reason": "missing_camera"})
            continue
        data = sparse_input_from_cached_batch(batch, resolver, intrinsics=camera.intrinsics, cfg=input_cfg)
        result = run_sparse_localization(data, loc_cfg)
        rows.append(
            {
                "query_id": batch.query_id,
                "success": bool(result.success),
                "inlier_count": int(result.inlier_count),
                "selected_count": int(len(result.selected_landmark_ids)),
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
        "score_mode": str(args.score_mode),
        "candidate_artifact": artifact.summarize_candidate_availability(),
    }
    hyperparameters = {
        "max_queries": int(args.max_queries),
        "max_keypoints": int(args.max_keypoints),
        "score_mode": str(args.score_mode),
        "rerank_prefix_fraction": float(args.rerank_prefix_fraction),
        "solver_weight": float(args.solver_weight),
        "native_weight": float(args.native_weight),
        "reprojection_error_px": float(args.reprojection_error_px),
        "pnp_iterations": int(args.pnp_iterations),
    }
    manifest = build_smoke_manifest(
        scene=str(args.scene),
        split_name=str(args.split_name),
        command=command,
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
