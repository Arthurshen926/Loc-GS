#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.stdloc_native.unified_lff_export import build_unified_lff_map


ABLATIONS = {
    "native": {
        "gate_locability_blend": 0.0,
        "gate_transform": "identity",
        "apply_to_detector_scores": False,
        "apply_to_ply_locability": False,
    },
    "selector": {
        "gate_transform": "identity",
        "apply_to_detector_scores": True,
        "apply_to_ply_locability": True,
    },
    "both": {
        "gate_transform": "identity",
        "apply_to_detector_scores": True,
        "apply_to_ply_locability": True,
    },
    "uniform": {
        "gate_transform": "uniform",
        "apply_to_detector_scores": True,
        "apply_to_ply_locability": True,
    },
    "permuted": {
        "gate_transform": "permuted",
        "apply_to_detector_scores": True,
        "apply_to_ply_locability": True,
    },
    "inverted": {
        "gate_transform": "inverted",
        "apply_to_detector_scores": True,
        "apply_to_ply_locability": True,
    },
    "detector_only": {
        "gate_transform": "identity",
        "apply_to_detector_scores": True,
        "apply_to_ply_locability": False,
    },
    "locability_only": {
        "gate_transform": "identity",
        "apply_to_detector_scores": False,
        "apply_to_ply_locability": True,
    },
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export selector-only causality ablation STDLoc maps.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gate_locability_blend", type=float, default=0.05)
    parser.add_argument("--descriptor_mode", choices=("native", "checkpoint"), default="native")
    parser.add_argument(
        "--ablation",
        default="native,selector,uniform,permuted,inverted,detector_only,locability_only,both",
        help="Comma-separated ablations to export.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_overwrite", action="store_true")
    return parser


def _parse_ablations(value: str) -> list[str]:
    names = [part.strip() for part in str(value).split(",") if part.strip()]
    unknown = sorted(set(names) - set(ABLATIONS))
    if unknown:
        raise ValueError(f"unsupported ablation(s): {', '.join(unknown)}")
    return names


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for name in _parse_ablations(args.ablation):
        cfg = dict(ABLATIONS[name])
        blend = float(cfg.pop("gate_locability_blend", args.gate_locability_blend))
        manifest = build_unified_lff_map(
            source_map=args.source_map,
            output_map=output_root / name,
            checkpoint_path=args.checkpoint_path,
            overwrite=not bool(args.no_overwrite),
            gate_locability_blend=blend,
            descriptor_mode=args.descriptor_mode,
            gate_transform=cfg["gate_transform"],
            gate_transform_seed=int(args.seed),
            apply_to_detector_scores=bool(cfg["apply_to_detector_scores"]),
            apply_to_ply_locability=bool(cfg["apply_to_ply_locability"]),
            ablation_type=name,
        )
        manifests.append(manifest)
    summary_path = output_root / "selector_ablation_export_summary.json"
    summary_path.write_text(json.dumps({"exports": manifests}, indent=2), encoding="utf-8")
    print(f"[selector_ablation_export] wrote {len(manifests)} maps under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
