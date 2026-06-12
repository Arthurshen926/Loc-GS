#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.results_metrics import build_sparse_metrics_from_results, load_query_ids


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute internal sparse metrics from per-query result rows.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--query_ids", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal sparse results metrics")
    query_ids = load_query_ids(args.query_ids) if args.query_ids is not None else None
    summary, rows = build_sparse_metrics_from_results(
        results_path=args.results,
        scene=str(args.scene),
        split_name=split,
        query_ids=query_ids,
    )
    command = [sys.executable, "-m", "loc_gs.scripts.build_internal_sparse_metrics_from_results", *(argv or sys.argv[1:])]
    manifest = {
        "schema_version": "internal_sparse_results_metrics_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "metrics_recompute_only",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "source_results": str(args.results),
        "query_ids": None if args.query_ids is None else str(args.query_ids),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "filtered_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
