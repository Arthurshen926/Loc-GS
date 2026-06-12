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
from loc_gs.sparse.candidate_completion_plan import build_candidate_completion_plan
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
    coverage_dir: str | Path | None,
    query_ids: str | Path | None,
    base_candidate_artifact: str | Path | None,
    shard_size: int,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate completion plan manifest")
    return {
        "schema_version": "internal_candidate_completion_plan_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "candidate_completion_plan",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "coverage_dir": None if coverage_dir is None else str(coverage_dir),
        "query_ids": None if query_ids is None else str(query_ids),
        "base_candidate_artifact": None if base_candidate_artifact is None else str(base_candidate_artifact),
        "hyperparameters": {"shard_size": int(shard_size)},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an internal plan for missing candidate-cache query shards.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--coverage_dir", type=Path, default=None)
    parser.add_argument("--query_ids", type=Path, default=None)
    parser.add_argument("--base_candidate_artifact", type=Path, default=None)
    parser.add_argument("--shard_size", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal candidate completion plan")
    if (args.coverage_dir is None) == (args.query_ids is None):
        raise ValueError("exactly one of --coverage_dir or --query_ids is required")
    coverage_summary_path = None
    coverage_summary = {}
    if args.coverage_dir is not None:
        coverage_summary_path = args.coverage_dir / "metrics_summary.json"
        coverage_summary = json.loads(coverage_summary_path.read_text(encoding="utf-8"))
        missing_query_ids = load_query_ids(args.coverage_dir / "missing_query_ids.txt")
    else:
        missing_query_ids = load_query_ids(args.query_ids)
    summary, shards = build_candidate_completion_plan(
        scene=str(args.scene),
        split_name=split,
        missing_query_ids=missing_query_ids,
        shard_size=int(args.shard_size),
        base_candidate_artifact=args.base_candidate_artifact,
    )
    summary.update(
        {
            "coverage_dir": None if args.coverage_dir is None else str(args.coverage_dir),
            "query_ids": None if args.query_ids is None else str(args.query_ids),
            "coverage_requested_query_count": coverage_summary.get("requested_query_count"),
            "coverage_covered_query_count": coverage_summary.get("covered_query_count"),
            "coverage_missing_query_count": coverage_summary.get("missing_query_count"),
        }
    )
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_candidate_completion_plan",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        coverage_dir=args.coverage_dir,
        query_ids=args.query_ids,
        base_candidate_artifact=args.base_candidate_artifact,
        shard_size=int(args.shard_size),
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed" if coverage_summary.get("split_audit_status") in (None, "passed") else "unknown",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
        "coverage_summary": None if coverage_summary_path is None else str(coverage_summary_path),
        "query_ids": None if args.query_ids is None else str(args.query_ids),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidate_completion_plan.jsonl").open("w", encoding="utf-8") as handle:
        for shard in shards:
            handle.write(json.dumps(shard, sort_keys=True) + "\n")
    (args.output_dir / "missing_query_ids.txt").write_text(
        "\n".join(missing_query_ids) + ("\n" if missing_query_ids else ""),
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
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
