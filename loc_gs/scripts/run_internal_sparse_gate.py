#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact, write_candidate_batches_jsonl
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.gate import build_sparse_gate_comparison


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


def build_gate_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    candidate_artifact: str | Path,
    baseline_metrics: str | Path,
    candidate_metrics: str | Path,
    candidate_coverage_metrics: str | Path | None,
    candidate_scorer_metrics: str | Path | None,
    candidate_conflict_graph_metrics: str | Path | None,
    hyperparameters: Mapping[str, object],
    data_root: str | Path | None = None,
    map_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal sparse gate manifest")
    return {
        "schema_version": "internal_sparse_train_dev_gate_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_only",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(candidate_artifact),
        "baseline_metrics": str(baseline_metrics),
        "candidate_metrics": str(candidate_metrics),
        "candidate_coverage_metrics": None if candidate_coverage_metrics is None else str(candidate_coverage_metrics),
        "candidate_scorer_metrics": None if candidate_scorer_metrics is None else str(candidate_scorer_metrics),
        "candidate_conflict_graph_metrics": None
        if candidate_conflict_graph_metrics is None
        else str(candidate_conflict_graph_metrics),
        "hyperparameters": dict(hyperparameters),
        "data_root": None if data_root is None else str(data_root),
        "map_path": None if map_path is None else str(map_path),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an internal sparse train-dev gate from cached candidate artifacts.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--baseline_metrics", type=Path, required=True)
    parser.add_argument("--candidate_metrics", type=Path, required=True)
    parser.add_argument("--candidate_coverage_metrics", type=Path, default=None)
    parser.add_argument("--candidate_scorer_metrics", type=Path, default=None)
    parser.add_argument("--candidate_conflict_graph_metrics", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dense_target_cm", type=float, default=10.0)
    parser.add_argument("--max_artifact_rows", type=int, default=None)
    parser.add_argument("--max_export_batches", type=int, default=2)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--map_path", type=Path, default=None)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    command = [sys.executable, "-m", "loc_gs.scripts.run_internal_sparse_gate", *(argv or sys.argv[1:])]
    hyperparameters = {
        "dense_target_cm": float(args.dense_target_cm),
        "max_artifact_rows": None if args.max_artifact_rows is None else int(args.max_artifact_rows),
        "max_export_batches": int(args.max_export_batches),
        "candidate_coverage_metrics": None
        if args.candidate_coverage_metrics is None
        else str(args.candidate_coverage_metrics),
        "candidate_scorer_metrics": None
        if args.candidate_scorer_metrics is None
        else str(args.candidate_scorer_metrics),
        "candidate_conflict_graph_metrics": None
        if args.candidate_conflict_graph_metrics is None
        else str(args.candidate_conflict_graph_metrics),
    }
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=args.max_artifact_rows)
    comparison = build_sparse_gate_comparison(
        scene=str(args.scene),
        split_name=str(args.split_name),
        baseline_metrics_path=args.baseline_metrics,
        candidate_metrics_path=args.candidate_metrics,
        dense_target_cm=float(args.dense_target_cm),
        candidate_artifact=artifact,
        candidate_coverage_metrics_path=args.candidate_coverage_metrics,
        candidate_scorer_metrics_path=args.candidate_scorer_metrics,
        candidate_conflict_graph_metrics_path=args.candidate_conflict_graph_metrics,
    )
    manifest = build_gate_manifest(
        scene=str(args.scene),
        split_name=str(args.split_name),
        command=command,
        candidate_artifact=args.candidate_artifact,
        baseline_metrics=args.baseline_metrics,
        candidate_metrics=args.candidate_metrics,
        candidate_coverage_metrics=args.candidate_coverage_metrics,
        candidate_scorer_metrics=args.candidate_scorer_metrics,
        candidate_conflict_graph_metrics=args.candidate_conflict_graph_metrics,
        hyperparameters=hyperparameters,
        data_root=args.data_root,
        map_path=args.map_path,
        checkpoint_path=args.checkpoint_path,
    )
    summary = comparison.build_metrics_summary()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_candidate_batches_jsonl(
        artifact.batches,
        args.output_dir / "candidate_batches_preview.jsonl",
        max_batches=max(0, int(args.max_export_batches)),
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate artifact did not include split_audit"}
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
