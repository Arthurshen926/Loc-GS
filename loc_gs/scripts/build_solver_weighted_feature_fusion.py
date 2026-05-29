#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loc_gs.stdloc_native.solver_weighted_feature_fusion import solver_weighted_fusion_from_pair_cache


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build solver-weighted landmark descriptor fusion artifact.")
    parser.add_argument("--pair_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust_alpha", type=float, default=0.1)
    parser.add_argument("--reprojection_threshold_px", type=float, default=4.0)
    parser.add_argument("--min_cosine", type=float, default=-1.0)
    parser.add_argument("--min_observations_per_landmark", type=int, default=1)
    parser.add_argument("--min_native_cosine", type=float, default=0.95)
    parser.add_argument("--allow_test_split", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output = Path(args.output)
    result = solver_weighted_fusion_from_pair_cache(
        args.pair_cache,
        trust_alpha=float(args.trust_alpha),
        reprojection_threshold_px=float(args.reprojection_threshold_px),
        min_cosine=float(args.min_cosine),
        min_observations_per_landmark=int(args.min_observations_per_landmark),
        min_native_cosine=float(args.min_native_cosine),
        disallow_test_split=not bool(args.allow_test_split),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    manifest = {
        "artifact": str(output),
        "pair_cache": str(args.pair_cache),
        "metadata": result["metadata"],
        "descriptor_shape": list(result["descriptors"].shape),
        "base_descriptor_shape": list(result["base_descriptors"].shape),
        "gaussian_id_count": int(result["gaussian_ids"].numel()),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), **result["metadata"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
