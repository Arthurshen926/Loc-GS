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
from loc_gs.sparse.candidate_completion_runner import build_candidate_shards_from_completion_plan


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
    completion_plan: str | Path,
    query_feature_cache: str | Path,
    base_candidate_artifact: str | Path,
    output_dir: str | Path,
    topk: int,
    max_landmarks: int | None,
    landmark_chunk_size: int | None,
    include_base_landmark_desc: bool,
    max_shards: int | None,
    skip_empty_shards: bool,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate completion runner manifest")
    return {
        "schema_version": "internal_candidate_completion_runner_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "candidate_completion_from_query_cache",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "completion_plan": str(completion_plan),
        "query_feature_cache": str(query_feature_cache),
        "base_candidate_artifact": str(base_candidate_artifact),
        "output_dir": str(output_dir),
        "hyperparameters": {
            "topk": int(topk),
            "max_landmarks": None if max_landmarks is None else int(max_landmarks),
            "landmark_chunk_size": None if landmark_chunk_size is None else int(landmark_chunk_size),
            "include_base_landmark_desc": bool(include_base_landmark_desc),
            "max_shards": None if max_shards is None else int(max_shards),
            "skip_empty_shards": bool(skip_empty_shards),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build all internal candidate completion shards from an internal query feature cache."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--completion_plan", type=Path, required=True)
    parser.add_argument("--query_feature_cache", type=Path, required=True)
    parser.add_argument("--base_candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--max_landmarks", type=int, default=None)
    parser.add_argument("--landmark_chunk_size", type=int, default=None)
    parser.add_argument("--omit_base_landmark_desc", action="store_true")
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--skip_empty_shards", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal candidate completion runner")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_candidate_shards_from_query_cache",
        *(argv or sys.argv[1:]),
    ]
    summary = build_candidate_shards_from_completion_plan(
        completion_plan=args.completion_plan,
        query_feature_cache=args.query_feature_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        output_dir=args.output_dir,
        scene=str(args.scene),
        split_name=split,
        topk=int(args.topk),
        max_landmarks=args.max_landmarks,
        landmark_chunk_size=args.landmark_chunk_size,
        include_base_landmark_desc=not bool(args.omit_base_landmark_desc),
        max_shards=args.max_shards,
        skip_empty_shards=bool(args.skip_empty_shards),
    )
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        completion_plan=args.completion_plan,
        query_feature_cache=args.query_feature_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        output_dir=args.output_dir,
        topk=int(args.topk),
        max_landmarks=args.max_landmarks,
        landmark_chunk_size=args.landmark_chunk_size,
        include_base_landmark_desc=not bool(args.omit_base_landmark_desc),
        max_shards=args.max_shards,
        skip_empty_shards=bool(args.skip_empty_shards),
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
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
    (args.output_dir / "candidate_shard_paths.txt").write_text(
        "\n".join(str(path) for path in summary["output_artifacts"])
        + ("\n" if summary["output_artifacts"] else ""),
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_status(), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
