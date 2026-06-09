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

from loc_gs.stdloc_native.solver_weighted_feature_fusion import solver_weighted_fusion_from_pair_cache


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build solver-weighted landmark descriptor fusion artifact.")
    parser.add_argument("--pair_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust_alpha", type=float, default=0.1)
    parser.add_argument("--reprojection_threshold_px", type=float, default=4.0)
    parser.add_argument("--min_cosine", type=float, default=-1.0)
    parser.add_argument("--min_observations_per_landmark", type=int, default=1)
    parser.add_argument("--min_native_cosine", type=float, default=0.95)
    parser.add_argument("--active_fusion_plan", default=None)
    parser.add_argument("--allow_test_split", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output = Path(args.output)
    repo_root = Path(__file__).resolve().parents[2]
    result = solver_weighted_fusion_from_pair_cache(
        args.pair_cache,
        trust_alpha=float(args.trust_alpha),
        reprojection_threshold_px=float(args.reprojection_threshold_px),
        min_cosine=float(args.min_cosine),
        min_observations_per_landmark=int(args.min_observations_per_landmark),
        min_native_cosine=float(args.min_native_cosine),
        disallow_test_split=not bool(args.allow_test_split),
        active_fusion_plan=args.active_fusion_plan,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    metadata = dict(result["metadata"])
    source_split = str(metadata.get("source_split_name", "unknown"))
    feedback_split = str(metadata.get("feedback_bank_split_name", "unknown"))
    test_split_used = source_split.lower() == "test" or feedback_split.lower() == "test"
    split_audit = {
        "schema_version": "solver_weighted_feature_fusion_split_audit_v1",
        "source_split_name": source_split,
        "feedback_bank_split_name": feedback_split,
        "test_split_used": bool(test_split_used),
        "official_test_used": bool(test_split_used),
        "role": "offline_solver_weighted_feature_fusion",
    }
    metrics = {
        **metadata,
        "artifact": str(output),
        "pair_cache": str(args.pair_cache),
        "active_fusion_plan": None if args.active_fusion_plan is None else str(args.active_fusion_plan),
        "descriptor_shape": list(result["descriptors"].shape),
        "base_descriptor_shape": list(result["base_descriptors"].shape),
        "gaussian_id_count": int(result["gaussian_ids"].numel()),
        "split_audit": split_audit,
    }
    manifest = {
        "schema_version": "solver_weighted_feature_fusion_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv if argv is None else [sys.executable, "-m", "loc_gs.scripts.build_solver_weighted_feature_fusion", *argv],
        "artifact": str(output),
        "pair_cache": str(args.pair_cache),
        "active_fusion_plan": None if args.active_fusion_plan is None else str(args.active_fusion_plan),
        "metadata": metadata,
        "descriptor_shape": list(result["descriptors"].shape),
        "base_descriptor_shape": list(result["base_descriptors"].shape),
        "gaussian_id_count": int(result["gaussian_ids"].numel()),
        "split_audit": split_audit,
        "official_test_used": bool(test_split_used),
    }
    manifest_path = output.with_suffix(".manifest.json")
    _write_json(manifest_path, manifest)
    _write_json(output.parent / "manifest.json", manifest)
    _write_json(output.parent / "metrics_summary.json", metrics)
    _write_json(output.parent / "split_audit.json", split_audit)
    (output.parent / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output.parent / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **result["metadata"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
