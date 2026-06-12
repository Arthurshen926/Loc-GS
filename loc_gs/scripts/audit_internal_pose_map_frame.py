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
from loc_gs.sparse.pose_map_frame_audit import audit_pose_map_frame


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


def build_pose_map_frame_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    candidate_artifact: str | Path,
    point_cloud: str | Path,
    cameras_json: str | Path,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="pose/map frame calibration audit manifest")
    return {
        "schema_version": "internal_pose_map_frame_audit_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "pose_map_frame_calibration",
        "inference_stage": "audit_only",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(candidate_artifact),
        "point_cloud": str(point_cloud),
        "cameras_json": str(cameras_json),
        "hyperparameters": hyperparameters,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit cached sparse pair pose/map/camera frame calibration.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=1024)
    parser.add_argument("--agreement_threshold_px", type=float, default=0.05)
    parser.add_argument("--min_positive_count", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = reject_test_split(str(args.split_name), purpose="pose/map frame calibration audit")
    if (args.image_width is None) != (args.image_height is None):
        raise ValueError("--image_width and --image_height must be provided together")
    command = [sys.executable, "-m", "loc_gs.scripts.audit_internal_pose_map_frame", *(argv or sys.argv[1:])]
    summary = audit_pose_map_frame(
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        target_width=args.image_width,
        target_height=args.image_height,
        max_rows=args.max_rows,
        agreement_threshold_px=args.agreement_threshold_px,
        min_positive_count=args.min_positive_count,
    )
    hyperparameters = {
        "image_width": args.image_width,
        "image_height": args.image_height,
        "max_rows": int(args.max_rows),
        "agreement_threshold_px": float(args.agreement_threshold_px),
        "min_positive_count": int(args.min_positive_count),
    }
    manifest = build_pose_map_frame_manifest(
        scene=str(args.scene),
        split_name=split_name,
        command=command,
        candidate_artifact=args.candidate_artifact,
        point_cloud=args.point_cloud,
        cameras_json=args.cameras_json,
        hyperparameters=hyperparameters,
    )
    split_audit = {
        "schema_version": "internal_pose_map_frame_split_audit_v1",
        "split_name": split_name,
        "status": "non_test_split_checked",
        "official_test_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "frame_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
