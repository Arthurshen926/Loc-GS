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
from loc_gs.training.sparse_candidate_scorer import (
    CandidateScorerConfig,
    classify_feature_input_policy,
    train_candidate_scorer,
)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an internal sparse candidate scorer from cached top-k labels.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--rank_feature_scale", type=float, default=1.0)
    parser.add_argument(
        "--feature_names",
        default=",".join(CandidateScorerConfig.feature_names),
        help="Comma-separated candidate scorer features.",
    )
    return parser


def _feature_names(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise ValueError("feature_names must contain at least one feature")
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal sparse candidate scorer training")
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=args.max_rows)
    cfg = CandidateScorerConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        rank_feature_scale=float(args.rank_feature_scale),
        feature_names=_feature_names(str(args.feature_names)),
    )
    feature_policy = classify_feature_input_policy(cfg.feature_names)
    model, summary = train_candidate_scorer(artifact, cfg)
    manifest = {
        "schema_version": "internal_sparse_candidate_scorer_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.train_internal_sparse_candidate_scorer", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_candidate_scorer_training",
        "dense_teacher_enabled": bool(summary.get("dense_teacher_sample_count", 0)),
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        **feature_policy,
        "candidate_artifact": str(args.candidate_artifact),
        "hyperparameters": {
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "rank_feature_scale": float(args.rank_feature_scale),
            "reprojection_error_scale_px": float(cfg.reprojection_error_scale_px),
            "protected_support_weight": float(cfg.protected_support_weight),
            "positive_inlier_weight": float(cfg.positive_inlier_weight),
            "hard_negative_weight": float(cfg.hard_negative_weight),
            "neutral_weight": float(cfg.neutral_weight),
            "feature_names": list(cfg.feature_names),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
