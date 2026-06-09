#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.sparse_pnp_validation import build_sparse_pnp_validation_profile


def _command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, "-m", "loc_gs.scripts.build_sparse_pnp_validation_profile", *sys.argv[1:]])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sparse-only self-map/train-dev PnP validation profile for SparseSet selection."
    )
    parser.add_argument("--baseline_run_dir", required=True, type=Path)
    parser.add_argument("--candidate_run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--baseline_feedback_bank", default=None, type=Path)
    parser.add_argument("--candidate_feedback_bank", default=None, type=Path)
    parser.add_argument("--protected_te_cm", default=15.0, type=float)
    parser.add_argument("--hard_te_cm", default=20.0, type=float)
    parser.add_argument("--regression_margin_cm", default=20.0, type=float)
    parser.add_argument("--improvement_margin_cm", default=20.0, type=float)
    parser.add_argument("--protect_all_baseline_good_queries", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    profile = build_sparse_pnp_validation_profile(
        baseline_run_dir=args.baseline_run_dir,
        candidate_run_dir=args.candidate_run_dir,
        scene=str(args.scene),
        split_name=str(args.split_name),
        protected_te_cm=float(args.protected_te_cm),
        hard_te_cm=float(args.hard_te_cm),
        regression_margin_cm=float(args.regression_margin_cm),
        improvement_margin_cm=float(args.improvement_margin_cm),
        baseline_feedback_bank=args.baseline_feedback_bank,
        candidate_feedback_bank=args.candidate_feedback_bank,
        protect_all_baseline_good_queries=bool(args.protect_all_baseline_good_queries),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "sparse_pnp_validation_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = dict(profile.get("metrics", {}))
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": str(args.split_name),
        "recipe": "sparse_set_v3_sparse_pnp_validation",
        "artifact": str(profile_path),
        "single_path_deployment": True,
        "branch_selection": False,
        "feedback_metadata": dict(profile.get("feedback_metadata", {})),
        "protected_feedback_metadata": dict(profile.get("protected_feedback_metadata", {})),
        "validated_feedback_metadata": dict(profile.get("validated_feedback_metadata", {})),
    }
    manifest = {
        "method": "loc_gs_sparse_pnp_validation_profile",
        **metadata,
        "baseline_run_dir": str(args.baseline_run_dir),
        "candidate_run_dir": str(args.candidate_run_dir),
        "baseline_feedback_bank": None if args.baseline_feedback_bank is None else str(args.baseline_feedback_bank),
        "candidate_feedback_bank": None if args.candidate_feedback_bank is None else str(args.candidate_feedback_bank),
        "profile_path": str(profile_path),
        "thresholds": profile.get("thresholds", {}),
    }
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"profile": str(profile_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
