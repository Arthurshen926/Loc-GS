#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact, write_candidate_batches_jsonl
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.candidate_shard_builder import build_candidate_shard_artifact, load_completion_shard


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
    completion_shard: str | Path,
    query_feature_cache: str | Path,
    base_candidate_artifact: str | Path,
    output_artifact: str | Path,
    topk: int,
    max_landmarks: int | None,
    landmark_chunk_size: int | None,
    query_chunk_size: int | None,
    include_base_landmark_desc: bool,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal candidate shard artifact manifest")
    return {
        "schema_version": "internal_candidate_shard_artifact_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "candidate_artifact_generation",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "completion_shard": str(completion_shard),
        "query_feature_cache": str(query_feature_cache),
        "base_candidate_artifact": str(base_candidate_artifact),
        "output_artifact": str(output_artifact),
        "hyperparameters": {
            "topk": int(topk),
            "max_landmarks": None if max_landmarks is None else int(max_landmarks),
            "landmark_chunk_size": None if landmark_chunk_size is None else int(landmark_chunk_size),
            "query_chunk_size": None if query_chunk_size is None else int(query_chunk_size),
            "include_base_landmark_desc": bool(include_base_landmark_desc),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an internal listwise candidate shard artifact from query features.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--completion_shard", type=Path, required=True)
    parser.add_argument("--shard_id", default=None)
    parser.add_argument("--query_feature_cache", type=Path, required=True)
    parser.add_argument("--base_candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="candidate_shard.pt")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_landmarks", type=int, default=None)
    parser.add_argument("--landmark_chunk_size", type=int, default=None)
    parser.add_argument("--query_chunk_size", type=int, default=None)
    parser.add_argument("--omit_base_landmark_desc", action="store_true")
    parser.add_argument("--max_export_batches", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal candidate shard artifact")
    shard = load_completion_shard(args.completion_shard, shard_id=args.shard_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_artifact = args.output_dir / str(args.output_name)
    summary = build_candidate_shard_artifact(
        completion_shard=shard,
        query_feature_cache=args.query_feature_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        output_artifact=output_artifact,
        scene=str(args.scene),
        split_name=split,
        topk=int(args.topk),
        max_landmarks=args.max_landmarks,
        landmark_chunk_size=args.landmark_chunk_size,
        query_chunk_size=args.query_chunk_size,
        include_base_landmark_desc=not bool(args.omit_base_landmark_desc),
    )
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_candidate_shard_artifact",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        completion_shard=args.completion_shard,
        query_feature_cache=args.query_feature_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        output_artifact=output_artifact,
        topk=int(args.topk),
        max_landmarks=args.max_landmarks,
        landmark_chunk_size=args.landmark_chunk_size,
        query_chunk_size=args.query_chunk_size,
        include_base_landmark_desc=not bool(args.omit_base_landmark_desc),
    )
    artifact = load_listwise_candidate_artifact(output_artifact, max_rows=1)
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate shard artifact did not include split_audit"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_candidate_batches_jsonl(
        load_listwise_candidate_artifact(output_artifact).batches,
        args.output_dir / "candidate_batches_preview.jsonl",
        max_batches=max(0, int(args.max_export_batches)),
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
