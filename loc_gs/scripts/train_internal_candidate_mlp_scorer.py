#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.sparse.audit import reject_test_split
from loc_gs.students.candidate_mlp_scorer import (
    CandidateMLPScorerConfig,
    train_candidate_mlp_scorer,
)
from loc_gs.training.sparse_candidate_scorer import classify_feature_input_policy


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an internal tensorized MLP sparse candidate scorer.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--listwise_loss_weight", type=float, default=1.0)
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
    split = reject_test_split(str(args.split_name), purpose="internal candidate MLP scorer training")
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=args.max_rows)
    scalar_feature_names = _feature_names(str(args.scalar_feature_names))
    cfg = CandidateMLPScorerConfig(
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        hidden_dim=int(args.hidden_dim),
        seed=int(args.seed),
        listwise_loss_weight=float(args.listwise_loss_weight),
        rank_feature_scale=float(args.rank_feature_scale),
        scalar_feature_names=scalar_feature_names,
    )
    feature_policy = classify_feature_input_policy(cfg.scalar_feature_names)
    model, summary = train_candidate_mlp_scorer(artifact, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pt"
    torch.save(model.to_torch_dict(), model_path)
    manifest = {
        "schema_version": "internal_candidate_mlp_scorer_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.train_internal_candidate_mlp_scorer", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_candidate_mlp_scorer_training",
        "dense_teacher_enabled": bool(summary.get("dense_teacher_sample_count", 0)),
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        **feature_policy,
        "candidate_artifact": str(args.candidate_artifact),
        "candidate_mlp_scorer": str(model_path),
        "hyperparameters": {
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "hidden_dim": int(args.hidden_dim),
            "seed": int(args.seed),
            "listwise_loss_weight": float(args.listwise_loss_weight),
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
