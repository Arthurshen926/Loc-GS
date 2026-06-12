#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from loc_gs.data.superpoint_query_extractor import extract_superpoint_query_feature_cache_for_queries
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.results_metrics import load_query_ids
from loc_gs.scripts.extract_internal_superpoint_features_for_queries import load_superpoint_model


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
    data_root: str | Path,
    query_ids: str | Path,
    output_cache: str | Path,
    weights: str | Path,
    batch_size: int,
    device: str,
    amp: bool,
    allow_missing: bool,
    max_keypoints: int,
    score_threshold: float | None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal query feature cache from images manifest")
    return {
        "schema_version": "internal_query_feature_cache_from_images_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "query_feature_cache_generation_from_images",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "data_root": str(data_root),
        "query_ids": str(query_ids),
        "output_cache": str(output_cache),
        "weights": str(weights),
        "hyperparameters": {
            "batch_size": int(batch_size),
            "device": str(device),
            "amp": bool(amp),
            "allow_missing": bool(allow_missing),
            "max_keypoints": int(max_keypoints),
            "score_threshold": None if score_threshold is None else float(score_threshold),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an internal sparse query feature cache directly from query images."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--query_ids", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", default="query_features.pt")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--max_keypoints", type=int, default=2048)
    parser.add_argument("--score_threshold", type=float, default=None)
    return parser


def _resolve_device(raw_device: str) -> torch.device:
    requested = str(raw_device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal query feature cache from images")
    query_ids = load_query_ids(args.query_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_cache = args.output_dir / str(args.output_name)
    device = _resolve_device(str(args.device))
    model = load_superpoint_model(weights=args.weights, device=device)
    summary = extract_superpoint_query_feature_cache_for_queries(
        scene=str(args.scene),
        split_name=split,
        data_root=args.data_root,
        query_ids=query_ids,
        output_cache=output_cache,
        model=model,
        device=device,
        max_keypoints=int(args.max_keypoints),
        score_threshold=args.score_threshold,
        batch_size=int(args.batch_size),
        strict_missing=not bool(args.allow_missing),
        amp=bool(args.amp),
    )
    command = [
        sys.executable,
        "-m",
        "loc_gs.scripts.build_internal_query_feature_cache_from_images",
        *(argv or sys.argv[1:]),
    ]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        data_root=args.data_root,
        query_ids=args.query_ids,
        output_cache=output_cache,
        weights=args.weights,
        batch_size=int(args.batch_size),
        device=str(device),
        amp=bool(args.amp),
        allow_missing=bool(args.allow_missing),
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
