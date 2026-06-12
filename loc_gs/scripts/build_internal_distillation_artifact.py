#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from loc_gs.sparse.audit import reject_test_split
from loc_gs.teacher.distillation_artifact import (
    DistillationArtifactConfig,
    build_distillation_payload,
    load_solver_feedback_label_rows,
)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an internal sparse-dense distillation candidate artifact.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--solver_feedback_labels", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dense_consistency_reprojection_px", type=float, default=4.0)
    parser.add_argument("--sparse_inlier_reprojection_px", type=float, default=8.0)
    parser.add_argument("--hard_negative_reprojection_px", type=float, default=8.0)
    parser.add_argument("--max_solver_weight", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="distillation artifact CLI")
    cfg = DistillationArtifactConfig(
        dense_consistency_reprojection_px=float(args.dense_consistency_reprojection_px),
        sparse_inlier_reprojection_px=float(args.sparse_inlier_reprojection_px),
        hard_negative_reprojection_px=float(args.hard_negative_reprojection_px),
        max_solver_weight=float(args.max_solver_weight),
    )
    payload = torch.load(args.candidate_artifact, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"candidate artifact must contain a dict payload: {args.candidate_artifact}")
    distilled, summary = build_distillation_payload(
        payload,
        load_solver_feedback_label_rows(args.solver_feedback_labels),
        scene=str(args.scene),
        split_name=split,
        cfg=cfg,
    )
    manifest = {
        "schema_version": "internal_distillation_artifact_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.build_internal_distillation_artifact", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "distillation_artifact_generation",
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(args.candidate_artifact),
        "solver_feedback_labels": str(args.solver_feedback_labels),
        "hyperparameters": {
            "dense_consistency_reprojection_px": float(args.dense_consistency_reprojection_px),
            "sparse_inlier_reprojection_px": float(args.sparse_inlier_reprojection_px),
            "hard_negative_reprojection_px": float(args.hard_negative_reprojection_px),
            "max_solver_weight": float(args.max_solver_weight),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(distilled, args.output_dir / "distilled_candidates.pt")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
