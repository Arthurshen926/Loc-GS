#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.candidate_coverage import (
    audit_candidate_artifact_coverage,
    load_requested_query_ids_from_results,
)
from loc_gs.sparse.results_metrics import load_query_ids


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    candidate_artifact: str | Path,
    query_ids: str | Path | None,
    query_results: str | Path | None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate coverage manifest")
    return {
        "schema_version": "internal_candidate_coverage_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "coverage_audit_only",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(candidate_artifact),
        "query_ids": None if query_ids is None else str(query_ids),
        "query_results": None if query_results is None else str(query_results),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit cached candidate artifact coverage for an internal sparse split.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--query_ids", type=Path, default=None)
    parser.add_argument("--query_results", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal candidate coverage audit")
    if (args.query_ids is None) == (args.query_results is None):
        raise ValueError("provide exactly one of --query_ids or --query_results")
    requested = (
        load_query_ids(args.query_ids)
        if args.query_ids is not None
        else load_requested_query_ids_from_results(args.query_results)
    )
    summary, details = audit_candidate_artifact_coverage(
        candidate_artifact=args.candidate_artifact,
        scene=str(args.scene),
        split_name=split,
        requested_query_ids=requested,
    )
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.audit_internal_candidate_coverage",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        candidate_artifact=args.candidate_artifact,
        query_ids=args.query_ids,
        query_results=args.query_results,
    )
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=1)
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate artifact did not include split_audit"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "coverage_details.json").write_text(
        json.dumps(details.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "covered_query_ids.txt").write_text(
        "\n".join(details.covered_query_ids) + ("\n" if details.covered_query_ids else ""),
        encoding="utf-8",
    )
    (args.output_dir / "missing_query_ids.txt").write_text(
        "\n".join(details.missing_query_ids) + ("\n" if details.missing_query_ids else ""),
        encoding="utf-8",
    )
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
