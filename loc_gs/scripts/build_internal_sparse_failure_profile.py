#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.failure_profile import SparseFailureProfileConfig, build_sparse_failure_profile


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


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_json_array(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON array: {path}")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"expected result row {idx} to be a JSON object: {path}")
        out.append(row)
    return out


def build_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    cached_eval_dir: str | Path,
    hyperparameters: dict[str, object],
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal sparse failure profile manifest")
    return {
        "schema_version": "internal_sparse_failure_profile_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_failure_profile",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "cached_eval_dir": str(cached_eval_dir),
        "hyperparameters": dict(hyperparameters),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a failure profile from internal cached sparse eval outputs.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--cached_eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--selected_correct_ratio_floor", type=float, default=0.10)
    parser.add_argument("--inlier_correct_ratio_floor", type=float, default=0.50)
    parser.add_argument("--target_median_te_cm", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal sparse failure profile")
    metrics_path = args.cached_eval_dir / "metrics_summary.json"
    results_path = args.cached_eval_dir / "results.json"
    metrics = _load_json_object(metrics_path)
    rows = _load_json_array(results_path)
    cfg = SparseFailureProfileConfig(
        selected_correct_ratio_floor=float(args.selected_correct_ratio_floor),
        inlier_correct_ratio_floor=float(args.inlier_correct_ratio_floor),
        target_median_te_cm=float(args.target_median_te_cm),
    )
    profile = build_sparse_failure_profile(rows, metrics=metrics, scene=str(args.scene), split_name=split, cfg=cfg)
    command = [sys.executable, "-m", "loc_gs.scripts.build_internal_sparse_failure_profile", *(argv or sys.argv[1:])]
    hyperparameters = {
        "selected_correct_ratio_floor": float(args.selected_correct_ratio_floor),
        "inlier_correct_ratio_floor": float(args.inlier_correct_ratio_floor),
        "target_median_te_cm": float(args.target_median_te_cm),
    }
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        cached_eval_dir=args.cached_eval_dir,
        hyperparameters=hyperparameters,
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failure_profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
