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

from loc_gs.diagnostics.ulfloc_detector_target_audit import (
    compare_detector_target_maps,
    sparse_result_error_map,
)


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_detector_targets(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"detector target artifact must be a dict: {path}")
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    split_audit = payload.get("split_audit", {})
    audit_split = str(split_audit.get("split_name", "")).strip() if isinstance(split_audit, Mapping) else ""
    if split.lower() == "test" or audit_split.lower() == "test":
        raise ValueError("refusing to audit detector targets from test split")
    if isinstance(split_audit, Mapping) and bool(split_audit.get("test_split_used", False)):
        raise ValueError("refusing to audit detector targets marked as test_split_used")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping):
        raise KeyError("detector target artifact must contain a targets mapping")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit two ULF-Loc scene-detector target artifacts.")
    parser.add_argument("--baseline_detector_targets", required=True, type=Path)
    parser.add_argument("--candidate_detector_targets", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--grid_size", default=8, type=int)
    parser.add_argument("--baseline_results", default=None, type=Path)
    parser.add_argument("--candidate_results", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_detector_targets(Path(args.baseline_detector_targets))
    candidate = _load_detector_targets(Path(args.candidate_detector_targets))
    baseline_errors = (
        sparse_result_error_map(_load_json(Path(args.baseline_results))) if args.baseline_results is not None else None
    )
    candidate_errors = (
        sparse_result_error_map(_load_json(Path(args.candidate_results))) if args.candidate_results is not None else None
    )
    rows, metrics = compare_detector_target_maps(
        baseline["targets"],
        candidate["targets"],
        height=int(args.height),
        width=int(args.width),
        grid_size=int(args.grid_size),
        baseline_errors=baseline_errors,
        candidate_errors=candidate_errors,
    )
    metrics = {
        "schema_version": "ulfloc_detector_target_audit_metrics_v1",
        **metrics,
        "height": int(args.height),
        "width": int(args.width),
        "grid_size": int(args.grid_size),
        "baseline_detector_targets": str(args.baseline_detector_targets),
        "candidate_detector_targets": str(args.candidate_detector_targets),
    }
    split_audit = {
        "schema_version": "ulfloc_detector_target_audit_split_audit_v1",
        "audit_status": "passed",
        "baseline_split_audit": baseline.get("split_audit", {}),
        "candidate_split_audit": candidate.get("split_audit", {}),
    }
    manifest = {
        "schema_version": "ulfloc_detector_target_audit_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "baseline_detector_targets": str(args.baseline_detector_targets),
        "candidate_detector_targets": str(args.candidate_detector_targets),
        "baseline_results": None if args.baseline_results is None else str(args.baseline_results),
        "candidate_results": None if args.candidate_results is None else str(args.candidate_results),
        "output_dir": str(output_dir),
        "metrics": metrics,
        "split_audit": split_audit,
        "paper_safe_role": "train_selfmap_detector_target_diagnostic",
    }
    _write_json(output_dir / "target_audit_rows.json", rows)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
