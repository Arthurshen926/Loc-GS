#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from loc_gs.eval.query_partitions import partitioned_stage_deltas
from loc_gs.stdloc_native.results import load_stdloc_query_results


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize hard/easy query partitions against a baseline run.")
    parser.add_argument("--baseline_run", required=True)
    parser.add_argument("--candidate_run", required=True)
    parser.add_argument("--candidate_name", default="")
    parser.add_argument("--stage", choices=("sparse", "dense"), default="dense")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    return parser


def _markdown_row(candidate: str, stage: str, partition: str, stats: dict[str, Any]) -> str:
    return (
        f"| {candidate} | {stage} | {partition} | {int(stats['query_count'])} | "
        f"{float(stats['mean_delta_te_cm']):.4f} | {float(stats['median_delta_te_cm']):.4f} | "
        f"{float(stats['recall_5cm_5deg_delta']):+.4f} | {float(stats['recall_2cm_2deg_delta']):+.4f} |"
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Query Partition Delta Report",
        "",
        f"- baseline: `{payload['baseline_run']}`",
        f"- candidate: `{payload['candidate_run']}`",
        f"- stage: `{payload['stage']}`",
        f"- git_commit: `{payload['git_commit']}`",
        "",
        "| candidate | stage | partition | queries | mean_delta_te_cm | median_delta_te_cm | r5_delta | r2_delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for partition, stats in payload["partitions"].items():
        lines.append(_markdown_row(payload["candidate"], payload["stage"], partition, stats))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    baseline_run = Path(args.baseline_run)
    candidate_run = Path(args.candidate_run)
    candidate = args.candidate_name or candidate_run.parent.name
    baseline_rows = load_stdloc_query_results(baseline_run)
    candidate_rows = load_stdloc_query_results(candidate_run)
    payload = {
        "protocol": "query_partition_delta_v1",
        "git_commit": _git_commit(),
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "candidate": str(candidate),
        "stage": str(args.stage),
        "partitions": partitioned_stage_deltas(baseline_rows, candidate_rows, stage=args.stage),
        "notes": {
            "partition_source": "baseline_stage_error_only",
            "test_split_forbidden_for_model_selection": True,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md), payload)
    print(json.dumps({"output_json": str(output_json), "output_md": args.output_md or None}, indent=2))


if __name__ == "__main__":
    main()
