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
from loc_gs.sparse.query_feature_cache import build_query_feature_cache_from_superpoint_dir
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
    superpoint_dir: str | Path,
    query_ids: str | Path | None,
    output_cache: str | Path,
    image_id_prefix: str,
    max_keypoints: int,
    score_threshold: float | None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal SuperPoint query feature cache manifest")
    return {
        "schema_version": "internal_query_superpoint_dir_cache_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "query_feature_cache_generation_from_superpoint_dir",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "superpoint_dir": str(superpoint_dir),
        "query_ids": None if query_ids is None else str(query_ids),
        "output_cache": str(output_cache),
        "image_id_prefix": str(image_id_prefix),
        "max_keypoints": int(max_keypoints),
        "score_threshold": None if score_threshold is None else float(score_threshold),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an internal query feature cache from a Loc-GS SuperPoint output directory."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--superpoint_dir", type=Path, required=True)
    parser.add_argument("--query_ids", type=Path, default=None)
    parser.add_argument("--image_id_prefix", default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="query_features.pt")
    parser.add_argument("--max_keypoints", type=int, default=2048)
    parser.add_argument("--score_threshold", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal query feature cache from SuperPoint dir")
    query_ids = load_query_ids(args.query_ids) if args.query_ids is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_cache = args.output_dir / str(args.output_name)
    summary = build_query_feature_cache_from_superpoint_dir(
        superpoint_dir=args.superpoint_dir,
        output_cache=output_cache,
        scene=str(args.scene),
        split_name=split,
        query_ids=query_ids,
        image_id_prefix=str(args.image_id_prefix),
        max_keypoints=int(args.max_keypoints),
        score_threshold=args.score_threshold,
    )
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_query_feature_cache_from_superpoint_dir",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        superpoint_dir=args.superpoint_dir,
        query_ids=args.query_ids,
        output_cache=output_cache,
        image_id_prefix=str(args.image_id_prefix),
        max_keypoints=int(args.max_keypoints),
        score_threshold=args.score_threshold,
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
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
