#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.sparse.audit import reject_test_split
from loc_gs.teacher.solver_feedback import (
    SolverFeedbackConfig,
    build_solver_feedback_labels,
    load_result_rows,
    summarize_solver_feedback_labels,
)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build internal sparse-dense solver feedback teacher labels.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--sparse_results", type=Path, required=True)
    parser.add_argument("--dense_results", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_dense_improvement_cm", type=float, default=2.0)
    parser.add_argument("--catastrophic_sparse_cm", type=float, default=50.0)
    parser.add_argument("--weight_clip_cm", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="solver feedback label CLI")
    cfg = SolverFeedbackConfig(
        min_dense_improvement_cm=float(args.min_dense_improvement_cm),
        catastrophic_sparse_cm=float(args.catastrophic_sparse_cm),
        weight_clip_cm=float(args.weight_clip_cm),
    )
    labels = build_solver_feedback_labels(
        scene=str(args.scene),
        split_name=split,
        sparse_rows=load_result_rows(args.sparse_results),
        dense_rows=load_result_rows(args.dense_results),
        cfg=cfg,
    )
    summary = summarize_solver_feedback_labels(labels)
    manifest = {
        "schema_version": "internal_solver_feedback_label_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.build_internal_solver_feedback_labels", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "training_label_generation",
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "sparse_results": str(args.sparse_results),
        "dense_results": str(args.dense_results),
        "hyperparameters": {
            "min_dense_improvement_cm": float(args.min_dense_improvement_cm),
            "catastrophic_sparse_cm": float(args.catastrophic_sparse_cm),
            "weight_clip_cm": float(args.weight_clip_cm),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label.to_json_dict(), sort_keys=True) + "\n")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
