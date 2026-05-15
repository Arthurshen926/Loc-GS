#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.stdloc_native.soft_prior import build_soft_prior_map


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a native STDLoc map with soft self-map reliability priors.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--calibrated_matchability_path", required=True)
    parser.add_argument("--base_cfg", default="third_party/stdloc/configs/stdloc_cambridge.yaml")
    parser.add_argument("--output_cfg", default="")
    parser.add_argument("--rho", type=float, default=-1.0)
    parser.add_argument("--selfmap_reliability_path", default="")
    parser.add_argument("--selfmap_reliability_stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--selfmap_reliability_center_cm", type=float, default=10.0)
    parser.add_argument("--selfmap_reliability_temperature_cm", type=float, default=1.0)
    parser.add_argument("--selfmap_reliability_r5_center", type=float, default=0.5)
    parser.add_argument("--selfmap_reliability_r5_temperature", type=float, default=0.1)
    parser.add_argument("--prior_blend", type=float, default=0.25)
    parser.add_argument("--fusion_mode", choices=["blend", "boost"], default="boost")
    parser.add_argument("--base_sparse_prior_weight", type=float, default=0.05)
    parser.add_argument("--base_dense_prior_weight", type=float, default=0.05)
    parser.add_argument("--no_point_cloud_locability", action="store_true")
    parser.add_argument("--no_overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    output_map = Path(args.output_map)
    output_cfg = Path(args.output_cfg) if args.output_cfg else output_map / "stdloc_soft_prior.yaml"
    manifest = build_soft_prior_map(
        source_map=Path(args.source_map),
        output_map=output_map,
        calibration_path=Path(args.calibrated_matchability_path),
        base_cfg_path=Path(args.base_cfg),
        output_cfg_path=output_cfg,
        rho=args.rho if args.rho >= 0.0 else None,
        selfmap_reliability_path=args.selfmap_reliability_path or None,
        selfmap_reliability_stage=args.selfmap_reliability_stage,
        selfmap_reliability_center_cm=args.selfmap_reliability_center_cm,
        selfmap_reliability_temperature_cm=args.selfmap_reliability_temperature_cm,
        selfmap_reliability_r5_center=(
            args.selfmap_reliability_r5_center if args.selfmap_reliability_r5_center >= 0.0 else None
        ),
        selfmap_reliability_r5_temperature=args.selfmap_reliability_r5_temperature,
        prior_blend=args.prior_blend,
        fusion_mode=args.fusion_mode,
        base_sparse_prior_weight=args.base_sparse_prior_weight,
        base_dense_prior_weight=args.base_dense_prior_weight,
        update_point_cloud_locability=not args.no_point_cloud_locability,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
