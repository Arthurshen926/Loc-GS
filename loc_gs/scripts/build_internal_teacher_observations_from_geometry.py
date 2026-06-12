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
from loc_gs.teacher.geometric_observations import build_teacher_observations_from_geometry


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    output_observations: str | Path,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="geometry teacher observation manifest")
    return {
        "schema_version": "internal_geometry_teacher_observation_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "training_teacher_observation_generation",
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(candidate_artifact),
        "point_cloud": str(point_cloud),
        "cameras_json": str(cameras_json),
        "output_observations": str(output_observations),
        "hyperparameters": hyperparameters,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build internal per-candidate teacher observations from geometry.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="teacher_observations.jsonl")
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--missing_principal_point", choices=["half_extent", "pixel_center"], default="pixel_center")
    parser.add_argument("--dense_consistency_reprojection_px", type=float, default=4.0)
    parser.add_argument("--sparse_inlier_reprojection_px", type=float, default=8.0)
    parser.add_argument("--hard_negative_reprojection_px", type=float, default=8.0)
    parser.add_argument("--max_solver_weight", type=float, default=4.0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--disable_frame_auto_calibration", action="store_true")
    parser.add_argument("--frame_calibration_max_rows", type=int, default=2048)
    parser.add_argument("--frame_calibration_agreement_threshold_px", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="geometry teacher observation CLI")
    observations, summary = build_teacher_observations_from_geometry(
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        scene=str(args.scene),
        split_name=split,
        target_width=args.target_width,
        target_height=args.target_height,
        missing_principal_point=str(args.missing_principal_point),
        dense_consistency_reprojection_px=float(args.dense_consistency_reprojection_px),
        sparse_inlier_reprojection_px=float(args.sparse_inlier_reprojection_px),
        hard_negative_reprojection_px=float(args.hard_negative_reprojection_px),
        max_solver_weight=float(args.max_solver_weight),
        max_rows=args.max_rows,
        auto_calibrate_frame=not bool(args.disable_frame_auto_calibration),
        frame_calibration_max_rows=int(args.frame_calibration_max_rows),
        frame_calibration_agreement_threshold_px=float(args.frame_calibration_agreement_threshold_px),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_observations = args.output_dir / str(args.output_name)
    with output_observations.open("w", encoding="utf-8") as handle:
        for row in observations:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    hyperparameters = {
        "target_width": args.target_width,
        "target_height": args.target_height,
        "missing_principal_point": str(args.missing_principal_point),
        "dense_consistency_reprojection_px": float(args.dense_consistency_reprojection_px),
        "sparse_inlier_reprojection_px": float(args.sparse_inlier_reprojection_px),
        "hard_negative_reprojection_px": float(args.hard_negative_reprojection_px),
        "max_solver_weight": float(args.max_solver_weight),
        "max_rows": None if args.max_rows is None else int(args.max_rows),
        "frame_auto_calibration": not bool(args.disable_frame_auto_calibration),
        "frame_calibration_max_rows": int(args.frame_calibration_max_rows),
        "frame_calibration_agreement_threshold_px": float(args.frame_calibration_agreement_threshold_px),
    }
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_teacher_observations_from_geometry",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        output_observations=output_observations,
        hyperparameters=hyperparameters,
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
