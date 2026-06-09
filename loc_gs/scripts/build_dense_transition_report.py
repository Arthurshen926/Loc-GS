#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loc_gs.dense_support.dense_transition_guard import (
    DenseTransitionGuardPolicy,
    build_dense_transition_report,
    load_dense_transition_records,
)


def _command(argv: Sequence[str] | None = None) -> str:
    parts = [sys.executable, "-m", "loc_gs.scripts.build_dense_transition_report"]
    parts.extend(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in parts)


def _git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - best-effort provenance
        return f"unavailable: {exc}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _policy_from_args(args: argparse.Namespace) -> DenseTransitionGuardPolicy:
    return DenseTransitionGuardPolicy(
        high_confidence_min_inliers=int(args.high_confidence_min_inliers),
        high_confidence_min_inlier_ratio=float(args.high_confidence_min_inlier_ratio),
        high_confidence_min_score=float(args.high_confidence_min_score),
        max_translation_delta_m=float(args.max_translation_delta_m),
        max_rotation_delta_deg=float(args.max_rotation_delta_deg),
        max_sparse_reprojection_error_px=float(args.max_sparse_reprojection_error_px),
        max_sparse_reprojection_worsening_px=float(args.max_sparse_reprojection_worsening_px),
        min_sparse_reprojection_retained_ratio=float(args.min_sparse_reprojection_retained_ratio),
        min_dense_valid_match_ratio=float(args.min_dense_valid_match_ratio),
        max_artifact_depth_risk_score=float(args.max_artifact_depth_risk_score),
        max_sparse_anchor_count=int(args.max_sparse_anchor_count),
        reject_weak_sparse_failures=bool(args.reject_weak_sparse_failures),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an inference-observable dense transition guard report."
    )
    parser.add_argument("--input", required=True, help="JSON list or {'results': [...]} records")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--allow_test_split", action="store_true")
    parser.add_argument("--checkpoint_path", default="unknown")
    parser.add_argument("--map_path", default="unknown")
    parser.add_argument("--data_root", default="unknown")
    parser.add_argument("--high_confidence_min_inliers", type=int, default=80)
    parser.add_argument("--high_confidence_min_inlier_ratio", type=float, default=0.0)
    parser.add_argument("--high_confidence_min_score", type=float, default=0.75)
    parser.add_argument("--max_translation_delta_m", type=float, default=0.35)
    parser.add_argument("--max_rotation_delta_deg", type=float, default=5.0)
    parser.add_argument("--max_sparse_reprojection_error_px", type=float, default=8.0)
    parser.add_argument("--max_sparse_reprojection_worsening_px", type=float, default=2.0)
    parser.add_argument("--min_sparse_reprojection_retained_ratio", type=float, default=0.90)
    parser.add_argument("--min_dense_valid_match_ratio", type=float, default=0.25)
    parser.add_argument("--max_artifact_depth_risk_score", type=float, default=0.65)
    parser.add_argument("--max_sparse_anchor_count", type=int, default=256)
    parser.add_argument("--reject_weak_sparse_failures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip() or "unknown"
    if split_name.lower() == "test" and not bool(args.allow_test_split):
        raise ValueError("test split dense transition reports are not allowed without --allow_test_split")

    policy = _policy_from_args(args)
    records = load_dense_transition_records(args.input)
    report = build_dense_transition_report(records, policy=policy, scene=str(args.scene), split_name=split_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    metrics_path = output_dir / "metrics_summary.json"
    manifest_path = output_dir / "manifest.json"
    command_path = output_dir / "command.txt"
    git_status_path = output_dir / "git_status.txt"

    _write_json(report_path, report)
    _write_json(metrics_path, report["summary"])
    command = _command(argv)
    command_path.write_text(command + "\n", encoding="utf-8")
    git_status_path.write_text(_git_text(["status", "--short"]) + "\n", encoding="utf-8")
    manifest = {
        "schema": "loc_gs_dense_transition_guard_manifest_v1",
        "method": "loc_gs_dense_transition_guard",
        "git_commit": _git_text(["rev-parse", "HEAD"]),
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scene": str(args.scene),
        "split": split_name,
        "split_name": split_name,
        "checkpoint_path": str(args.checkpoint_path),
        "map_path": str(args.map_path),
        "data_root": str(args.data_root),
        "input": str(args.input),
        "report_path": str(report_path),
        "metrics_summary_path": str(metrics_path),
        "git_status_path": str(git_status_path),
        "hyperparameters": report["policy"],
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "uses_gt": False,
        "inference_observable_only": True,
        "diagnostic_only": True,
        "paper_safe_main_method": False,
        "required_integration_hook": "dense_stats_and_sparse_inliers",
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "metrics_summary": str(metrics_path),
                **report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
