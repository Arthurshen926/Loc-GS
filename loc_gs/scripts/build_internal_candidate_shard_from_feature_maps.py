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
from loc_gs.sparse.candidate_shard_builder import build_candidate_shard_artifact, load_completion_shard
from loc_gs.sparse.query_feature_cache import build_query_feature_cache_from_feature_map_cache


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
    feature_map_cache: str | Path,
    base_candidate_artifact: str | Path,
    query_feature_cache: str | Path,
    output_artifact: str | Path,
    topk: int,
    max_keypoints: int,
    max_landmarks: int | None,
    score_threshold: float | None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="candidate shard from feature maps manifest")
    return {
        "schema_version": "internal_candidate_shard_from_feature_maps_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "candidate_completion_from_internal_feature_maps",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "completion_shard": str(completion_shard),
        "feature_map_cache": str(feature_map_cache),
        "base_candidate_artifact": str(base_candidate_artifact),
        "query_feature_cache": str(query_feature_cache),
        "output_artifact": str(output_artifact),
        "hyperparameters": {
            "topk": int(topk),
            "max_keypoints": int(max_keypoints),
            "max_landmarks": None if max_landmarks is None else int(max_landmarks),
            "score_threshold": None if score_threshold is None else float(score_threshold),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an internal candidate completion shard directly from dense feature-map cache."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--completion_shard", type=Path, required=True)
    parser.add_argument("--shard_id", default=None)
    parser.add_argument("--feature_map_cache", type=Path, required=True)
    parser.add_argument("--base_candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--query_cache_name", default="query_features.pt")
    parser.add_argument("--candidate_artifact_name", default="candidate_shard.pt")
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--max_keypoints", type=int, default=2048)
    parser.add_argument("--max_landmarks", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="candidate shard from feature maps CLI")
    shard = load_completion_shard(args.completion_shard, shard_id=args.shard_id)
    query_ids = [str(value) for value in shard.get("query_ids", [])]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_cache = args.output_dir / str(args.query_cache_name)
    output_artifact = args.output_dir / str(args.candidate_artifact_name)
    query_summary = build_query_feature_cache_from_feature_map_cache(
        feature_map_cache=args.feature_map_cache,
        output_cache=query_cache,
        scene=str(args.scene),
        split_name=split,
        query_ids=query_ids,
        max_keypoints=int(args.max_keypoints),
        score_threshold=args.score_threshold,
    )
    candidate_summary = build_candidate_shard_artifact(
        completion_shard=shard,
        query_feature_cache=query_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        output_artifact=output_artifact,
        scene=str(args.scene),
        split_name=split,
        topk=int(args.topk),
        max_landmarks=args.max_landmarks,
    )
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_candidate_shard_from_feature_maps",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        completion_shard=args.completion_shard,
        feature_map_cache=args.feature_map_cache,
        base_candidate_artifact=args.base_candidate_artifact,
        query_feature_cache=query_cache,
        output_artifact=output_artifact,
        topk=int(args.topk),
        max_keypoints=int(args.max_keypoints),
        max_landmarks=args.max_landmarks,
        score_threshold=args.score_threshold,
    )
    summary = {
        "schema_version": "internal_candidate_shard_from_feature_maps_summary_v1",
        "scene": str(args.scene),
        "split_name": split,
        "query_feature_summary": query_summary,
        "candidate_shard_summary": candidate_summary,
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }
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
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
