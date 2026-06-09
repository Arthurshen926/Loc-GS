#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from loc_gs.stdloc_native.solver_weighted_feature_fusion import (
    guard_solver_weighted_fusion_from_validation_profile,
)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guard solver-fused ULF-Loc landmark descriptors with train-dev sparse-PnP validation regressions."
    )
    parser.add_argument("--descriptor_artifact", required=True, type=Path)
    parser.add_argument("--pair_cache", required=True, type=Path)
    parser.add_argument("--validation_profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--regression_margin_cm", default=20.0, type=float)
    parser.add_argument("--improvement_margin_cm", default=20.0, type=float)
    parser.add_argument("--reprojection_threshold_px", default=None, type=float)
    parser.add_argument("--min_cosine", default=-1.0, type=float)
    parser.add_argument("--min_regression_support", default=1e-6, type=float)
    parser.add_argument("--improvement_credit_ratio", default=1.0, type=float)
    parser.add_argument(
        "--guard_query_scope",
        default="all_regressions",
        choices=["all_regressions", "protected_regressions"],
    )
    parser.add_argument("--allow_test_split", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = str(args.split_name).strip()
    if not split:
        raise ValueError("split_name is required")
    if split.lower() == "test" and not bool(args.allow_test_split):
        raise ValueError("test split is not allowed for guarded solver-fused descriptors")

    repo_root = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = guard_solver_weighted_fusion_from_validation_profile(
        args.descriptor_artifact,
        args.pair_cache,
        args.validation_profile,
        regression_margin_cm=float(args.regression_margin_cm),
        improvement_margin_cm=float(args.improvement_margin_cm),
        reprojection_threshold_px=args.reprojection_threshold_px,
        min_cosine=float(args.min_cosine),
        min_regression_support=float(args.min_regression_support),
        improvement_credit_ratio=float(args.improvement_credit_ratio),
        guard_query_scope=str(args.guard_query_scope),
        disallow_test_split=not bool(args.allow_test_split),
    )
    torch.save(result, output)

    metrics = dict(result.get("metadata", {}))
    split_audit = {
        "schema_version": "guarded_solver_weighted_feature_fusion_split_audit_v1",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": split.lower() == "test",
        "role": "train_dev_sparse_pnp_validation_guard",
    }
    manifest = {
        "schema_version": "guarded_solver_weighted_feature_fusion_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": [sys.executable, "-m", "loc_gs.scripts.guard_solver_weighted_feature_fusion", *(argv or sys.argv[1:])],
        "scene": str(args.scene),
        "split": split,
        "descriptor_artifact": str(args.descriptor_artifact),
        "pair_cache": str(args.pair_cache),
        "validation_profile": str(args.validation_profile),
        "output": str(output),
        "metrics_summary": metrics,
        "split_audit": split_audit,
        "official_test_used": False,
        "branch_selection": False,
        "single_path_deployment": True,
    }
    _write_json(output.parent / "manifest.json", manifest)
    _write_json(output.parent / "metrics_summary.json", metrics)
    _write_json(output.parent / "split_audit.json", split_audit)
    (output.parent / "command.txt").write_text(
        " ".join(shlex.quote(str(part)) for part in manifest["command"]) + "\n",
        encoding="utf-8",
    )
    (output.parent / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"output": str(output), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
