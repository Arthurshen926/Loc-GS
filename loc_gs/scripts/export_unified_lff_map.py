#!/usr/bin/env python3
from __future__ import annotations

import argparse

from loc_gs.stdloc_native.unified_lff_export import build_unified_lff_map


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Unified LFF-v2 checkpoint to a STDLoc-compatible map.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--gate_locability_blend", type=float, default=0.0)
    parser.add_argument("--descriptor_mode", choices=("checkpoint", "native"), default="checkpoint")
    parser.add_argument("--no_overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    manifest = build_unified_lff_map(
        source_map=args.source_map,
        output_map=args.output_map,
        checkpoint_path=args.checkpoint_path,
        gate_locability_blend=float(args.gate_locability_blend),
        descriptor_mode=args.descriptor_mode,
        overwrite=not bool(args.no_overwrite),
    )
    print(f"[unified_lff_export] wrote {manifest['output_map']}")


if __name__ == "__main__":
    main()
