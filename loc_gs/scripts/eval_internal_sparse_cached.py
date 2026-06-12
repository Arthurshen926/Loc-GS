#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.cached_eval import CachedSparseEvalConfig, run_cached_sparse_eval
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"unknown: {exc}\n"


def build_cached_eval_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    candidate_scorer: str | Path | None,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal sparse cached eval manifest")
    return {
        "schema_version": "internal_sparse_cached_eval_manifest_v1",
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
    parser = argparse.ArgumentParser(description="Evaluate internal sparse-only localization from cached candidates.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--missing_principal_point", choices=["half_extent", "pixel_center"], default="pixel_center")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_keypoints", type=int, default=1024)
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
    split = reject_test_split(str(args.split_name), purpose="internal sparse cached eval")
    command = [sys.executable, "-m", "loc_gs.scripts.eval_internal_sparse_cached", *(argv or sys.argv[1:])]
    cfg = CachedSparseEvalConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        missing_principal_point=str(args.missing_principal_point),
        max_queries=args.max_queries,
        max_keypoints=int(args.max_keypoints),
        score_mode=str(args.score_mode),
        candidate_scorer=args.candidate_scorer,
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
    summary, rows = run_cached_sparse_eval(
        scene=str(args.scene),
        split_name=split,
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        cfg=cfg,
    )
    hyperparameters = {
        "image_width": args.image_width,
        "image_height": args.image_height,
        "missing_principal_point": str(args.missing_principal_point),
        "max_queries": args.max_queries,
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
    manifest = build_cached_eval_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        candidate_scorer=args.candidate_scorer,
        hyperparameters=hyperparameters,
    )
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=1)
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate artifact did not include split_audit"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
