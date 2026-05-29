#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import torch

from loc_gs.dense_support.stable_region_mask import build_stable_region_proxy_mask
from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v11 no-dependency stable-region proxy mask artifact.")
    parser.add_argument("--image_tensor", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--multi_view_support_tensor", default="")
    parser.add_argument("--stable_threshold", type=float, default=0.5)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split_name = str(args.split_name).strip()
    if split_name.lower() == "test":
        raise ValueError("test split images are not allowed for stable-region mask selection artifacts")
    image = torch.load(Path(args.image_tensor), map_location="cpu")
    support = torch.load(Path(args.multi_view_support_tensor), map_location="cpu") if str(args.multi_view_support_tensor).strip() else None
    result = build_stable_region_proxy_mask(
        image,
        multi_view_support=support,
        stable_threshold=float(args.stable_threshold),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "stable_region_mask.pt"
    torch.save(result, artifact_path)
    stable_mask = result["stable_mask"]
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": split_name or "unknown",
        "recipe": "lsf_v11_stable_proxy_mask",
        "artifact": str(artifact_path),
        "external_model": "none",
        "single_path_deployment": True,
        "branch_selection": False,
    }
    metrics = {
        "height": int(stable_mask.shape[-2]),
        "width": int(stable_mask.shape[-1]),
        "stable_ratio": float(stable_mask.float().mean().item()) if stable_mask.numel() else 0.0,
        "risk_mean": float(result["risk"].float().mean().item()) if result["risk"].numel() else 0.0,
    }
    manifest = {
        "method": "loc_gs_lsf_v11_stable_region_proxy_mask",
        **metadata,
        "image_tensor": str(args.image_tensor),
        "multi_view_support_tensor": str(args.multi_view_support_tensor),
        "stable_threshold": float(args.stable_threshold),
        "mask_metadata": result["metadata"],
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
