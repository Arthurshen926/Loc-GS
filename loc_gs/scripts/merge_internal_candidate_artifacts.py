#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.candidate_artifact_merge import merge_listwise_candidate_artifacts


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


def build_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    input_artifacts: Sequence[str | Path],
    output_artifact: str | Path,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate artifact merge manifest")
    return {
        "schema_version": "internal_candidate_artifact_merge_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "candidate_artifact_merge",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "input_artifacts": [str(path) for path in input_artifacts],
        "output_artifact": str(output_artifact),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge internal listwise candidate artifacts into one eval cache.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--input_artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="merged_candidates.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal candidate artifact merge")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_artifact = args.output_dir / str(args.output_name)
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.merge_internal_candidate_artifacts",
        *(argv or sys.argv[1:]),
    ]
    summary = merge_listwise_candidate_artifacts(
        input_artifacts=args.input_artifacts,
        output_artifact=output_artifact,
        scene=str(args.scene),
        split_name=split,
    )
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        input_artifacts=args.input_artifacts,
        output_artifact=output_artifact,
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": summary["split_audit_status"],
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
