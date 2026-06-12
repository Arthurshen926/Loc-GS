#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.students.landmark_selector import LandmarkSelectorConfig, train_landmark_selector


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an internal sparse landmark selector from distilled candidates.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--protected_support_gain", type=float, default=2.0)
    parser.add_argument("--positive_inlier_gain", type=float, default=1.0)
    parser.add_argument("--dense_consistency_gain", type=float, default=0.5)
    parser.add_argument("--hard_negative_penalty", type=float, default=1.0)
    parser.add_argument("--conflict_penalty", type=float, default=0.1)
    parser.add_argument("--score_scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal landmark selector training")
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=args.max_rows)
    cfg = LandmarkSelectorConfig(
        protected_support_gain=float(args.protected_support_gain),
        positive_inlier_gain=float(args.positive_inlier_gain),
        dense_consistency_gain=float(args.dense_consistency_gain),
        hard_negative_penalty=float(args.hard_negative_penalty),
        conflict_penalty=float(args.conflict_penalty),
        score_scale=float(args.score_scale),
    )
    model, summary = train_landmark_selector(artifact.batches, cfg)
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate artifact did not include split_audit"}
    manifest = {
        "schema_version": "internal_landmark_selector_training_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.train_internal_landmark_selector", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_landmark_selector_training",
        "dense_teacher_enabled": bool(
            summary.get("protected_support_count", 0)
            or summary.get("positive_inlier_count", 0)
            or summary.get("hard_negative_count", 0)
        ),
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(args.candidate_artifact),
        "student_modules": ["landmark_selector", "conflict_graph"],
        "hyperparameters": {
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "protected_support_gain": float(args.protected_support_gain),
            "positive_inlier_gain": float(args.positive_inlier_gain),
            "dense_consistency_gain": float(args.dense_consistency_gain),
            "hard_negative_penalty": float(args.hard_negative_penalty),
            "conflict_penalty": float(args.conflict_penalty),
            "score_scale": float(args.score_scale),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "landmark_selector.json").write_text(
        json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conflict_graph.json").write_text(
        json.dumps(model.to_conflict_graph_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
