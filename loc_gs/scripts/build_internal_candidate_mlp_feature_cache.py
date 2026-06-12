#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.students.candidate_mlp_scorer import (
    CandidateMLPScorerConfig,
    build_candidate_mlp_feature_cache,
    save_candidate_mlp_feature_cache,
)
from loc_gs.training.sparse_candidate_scorer import classify_feature_input_policy


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reusable feature cache for the internal candidate MLP scorer.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Rows per vectorized feature chunk while constructing the cache.",
    )
    parser.add_argument("--rank_feature_scale", type=float, default=1.0)
    parser.add_argument(
        "--scalar_feature_names",
        default=",".join(CandidateMLPScorerConfig.scalar_feature_names),
        help="Comma-separated inference-safe scalar features prepended to descriptor-pair vectors.",
    )
    return parser


def _feature_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("scalar_feature_names must contain at least one feature")
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal candidate MLP feature cache build")
    scalar_feature_names = _feature_names(str(args.scalar_feature_names))
    cfg = CandidateMLPScorerConfig(
        batch_size=int(args.batch_size),
        rank_feature_scale=float(args.rank_feature_scale),
        scalar_feature_names=scalar_feature_names,
    )
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=args.max_rows)
    cache = build_candidate_mlp_feature_cache(artifact, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = save_candidate_mlp_feature_cache(cache, args.output_dir / "feature_cache.pt")
    summary = cache.summarize()
    feature_policy = classify_feature_input_policy(cfg.scalar_feature_names)
    manifest = {
        "schema_version": "internal_candidate_mlp_feature_cache_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_internal_candidate_mlp_feature_cache",
            *(argv or sys.argv[1:]),
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_candidate_mlp_feature_cache",
        "dense_teacher_enabled": bool(summary.get("dense_teacher_sample_count", 0)),
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        **feature_policy,
        "candidate_artifact": str(args.candidate_artifact),
        "feature_cache": str(cache_path),
        "hyperparameters": {
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "batch_size": int(args.batch_size),
            "rank_feature_scale": float(args.rank_feature_scale),
            "scalar_feature_names": list(cfg.scalar_feature_names),
        },
    }
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
