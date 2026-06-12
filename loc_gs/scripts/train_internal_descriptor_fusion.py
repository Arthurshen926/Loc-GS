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
from loc_gs.students.descriptor_fusion import DescriptorFusionConfig, train_descriptor_fusion_from_payload


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an internal sparse descriptor fusion student.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--candidate_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--trust_region", type=float, default=0.25)
    parser.add_argument("--protected_support_gain", type=float, default=2.0)
    parser.add_argument("--positive_inlier_gain", type=float, default=1.0)
    parser.add_argument("--hard_negative_penalty", type=float, default=1.0)
    parser.add_argument("--score_scale", type=float, default=1.0)
    parser.add_argument("--skip_json_model", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal descriptor fusion training")
    payload = torch.load(args.candidate_artifact, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"candidate artifact must contain a dict payload: {args.candidate_artifact}")
    artifact = load_listwise_candidate_artifact(args.candidate_artifact, max_rows=1)
    cfg = DescriptorFusionConfig(
        trust_region=float(args.trust_region),
        protected_support_gain=float(args.protected_support_gain),
        positive_inlier_gain=float(args.positive_inlier_gain),
        hard_negative_penalty=float(args.hard_negative_penalty),
        score_scale=float(args.score_scale),
    )
    model, summary = train_descriptor_fusion_from_payload(payload, cfg)
    split_audit = artifact.metadata.get("split_audit")
    if not isinstance(split_audit, dict):
        split_audit = {"audit_status": "unknown", "reason": "candidate artifact did not include split_audit"}
    manifest = {
        "schema_version": "internal_descriptor_fusion_training_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.train_internal_descriptor_fusion", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_descriptor_fusion_training",
        "dense_teacher_enabled": bool(
            summary.get("protected_support_count", 0)
            or summary.get("positive_inlier_count", 0)
            or summary.get("hard_negative_count", 0)
        ),
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "candidate_artifact": str(args.candidate_artifact),
        "descriptor_fusion_model": str(args.output_dir / "descriptor_fusion.pt"),
        "descriptor_fusion_json_model": None if args.skip_json_model else str(args.output_dir / "descriptor_fusion.json"),
        "student_modules": ["descriptor_fusion"],
        "hyperparameters": {
            "trust_region": float(args.trust_region),
            "protected_support_gain": float(args.protected_support_gain),
            "positive_inlier_gain": float(args.positive_inlier_gain),
            "hard_negative_penalty": float(args.hard_negative_penalty),
            "score_scale": float(args.score_scale),
            "skip_json_model": bool(args.skip_json_model),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.to_torch_dict(), args.output_dir / "descriptor_fusion.pt")
    if not bool(args.skip_json_model):
        (args.output_dir / "descriptor_fusion.json").write_text(
            json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n",
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
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
