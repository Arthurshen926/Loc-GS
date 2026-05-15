#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.stdloc_native.lff_export import build_lff_feature_map


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a native STDLoc map whose loc_* field contains a bounded LFF descriptor residual."
    )
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--calibrated_matchability_path", default="")
    parser.add_argument("--base_cfg", default="third_party/stdloc/configs/stdloc_cambridge.yaml")
    parser.add_argument("--output_cfg", default="")
    parser.add_argument("--rho", type=float, default=-1.0)
    parser.add_argument("--selfmap_reliability_path", default="")
    parser.add_argument("--selfmap_reliability_stage", choices=["sparse", "dense"], default="dense")
    parser.add_argument("--selfmap_reliability_center_cm", type=float, default=10.0)
    parser.add_argument("--selfmap_reliability_temperature_cm", type=float, default=1.0)
    parser.add_argument("--selfmap_reliability_r5_center", type=float, default=0.5)
    parser.add_argument("--selfmap_reliability_r5_temperature", type=float, default=0.1)
    parser.add_argument("--descriptor_alpha_max", type=float, default=0.03)
    parser.add_argument("--decode_chunk_size", type=int, default=16384)
    parser.add_argument(
        "--selector_mode",
        choices=["uniform", "locability", "matchability", "combined", "reliability_boost"],
        default="reliability_boost",
    )
    parser.add_argument("--selector_matchability_weight", type=float, default=1.0)
    parser.add_argument("--selector_source_weight", type=float, default=0.0)
    parser.add_argument("--selector_floor", type=float, default=0.0)
    parser.add_argument("--selector_power", type=float, default=1.0)
    parser.add_argument("--selector_locability_weight", type=float, default=0.0)
    parser.add_argument("--locability_fusion_mode", choices=["blend", "boost"], default="boost")
    parser.add_argument("--prior_blend", type=float, default=0.25)
    parser.add_argument("--score_fusion_mode", choices=["blend", "boost"], default="boost")
    parser.add_argument("--base_sparse_prior_weight", type=float, default=0.0)
    parser.add_argument("--base_dense_prior_weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    output_map = Path(args.output_map)
    output_cfg = Path(args.output_cfg) if args.output_cfg else output_map / "stdloc_lff.yaml"
    manifest = build_lff_feature_map(
        source_map=Path(args.source_map),
        output_map=output_map,
        checkpoint_path=Path(args.checkpoint_path),
        base_cfg_path=Path(args.base_cfg),
        output_cfg_path=output_cfg,
        calibration_path=Path(args.calibrated_matchability_path) if args.calibrated_matchability_path else None,
        rho=args.rho if args.rho >= 0.0 else None,
        selfmap_reliability_path=args.selfmap_reliability_path or None,
        selfmap_reliability_stage=args.selfmap_reliability_stage,
        selfmap_reliability_center_cm=args.selfmap_reliability_center_cm,
        selfmap_reliability_temperature_cm=args.selfmap_reliability_temperature_cm,
        selfmap_reliability_r5_center=(
            args.selfmap_reliability_r5_center if args.selfmap_reliability_r5_center >= 0.0 else None
        ),
        selfmap_reliability_r5_temperature=args.selfmap_reliability_r5_temperature,
        descriptor_alpha_max=args.descriptor_alpha_max,
        decode_chunk_size=args.decode_chunk_size,
        selector_mode=args.selector_mode,
        selector_matchability_weight=args.selector_matchability_weight,
        selector_source_weight=args.selector_source_weight,
        selector_floor=args.selector_floor,
        selector_power=args.selector_power,
        selector_locability_weight=args.selector_locability_weight,
        locability_fusion_mode=args.locability_fusion_mode,
        prior_blend=args.prior_blend,
        score_fusion_mode=args.score_fusion_mode,
        base_sparse_prior_weight=args.base_sparse_prior_weight,
        base_dense_prior_weight=args.base_dense_prior_weight,
        overwrite=not args.no_overwrite,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
