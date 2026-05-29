#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import torch

from loc_gs.dense_support.dense_match_verifier import verify_dense_matches
from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _field(payload: dict[str, Any], name: str) -> Any:
    if name not in payload:
        raise KeyError(f"dense verifier input missing {name}")
    return payload[name]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v10 dense match verifier mask/weight artifact.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("mask", "weight"), default="mask")
    parser.add_argument("--min_verifier_score", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--allow_test_split", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip()
    if split_name.lower() == "test" and not bool(args.allow_test_split):
        raise ValueError("test split dense verifier artifacts are not allowed without --allow_test_split")
    payload = torch.load(Path(args.input), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("dense verifier input must be a torch-saved dict")
    result = verify_dense_matches(
        query_yx=_field(payload, "query_yx"),
        reference_yx=_field(payload, "reference_yx"),
        descriptor_scores=_field(payload, "descriptor_scores"),
        local_geometry=_field(payload, "local_geometry"),
        lsf_support=_field(payload, "lsf_support"),
        alpha_dominance=_field(payload, "alpha_dominance"),
        ambiguity=_field(payload, "ambiguity"),
        depth_uncertainty=_field(payload, "depth_uncertainty"),
        pose_leverage=_field(payload, "pose_leverage"),
        min_verifier_score=float(args.min_verifier_score),
        topk=int(args.topk) if int(args.topk) > 0 else None,
        reweight_scores=str(args.mode) == "weight",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "dense_match_verifier.pt"
    torch.save(result, artifact_path)
    recipe = f"lsf_v10_dense_verifier_{str(args.mode)}"
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": split_name or "unknown",
        "recipe": recipe,
        "artifact": str(artifact_path),
        "single_path_deployment": True,
        "branch_selection": False,
        "paper_safe_for_tuning": split_name.lower() != "test",
    }
    metrics = dict(result["metadata"])
    manifest = {
        "method": "loc_gs_lsf_v10_dense_match_verifier",
        **metadata,
        "input": str(args.input),
        "mode": str(args.mode),
        "min_verifier_score": float(args.min_verifier_score),
        "topk": int(args.topk),
    }
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"artifact": str(artifact_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
