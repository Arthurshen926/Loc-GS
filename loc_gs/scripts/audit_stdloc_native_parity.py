#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.scripts.audit_cambridge_parity import audit_cambridge_parity


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit query-level parity between STDLoc-native output and official STDLoc output."
    )
    parser.add_argument("--native_dir", required=True)
    parser.add_argument("--stdloc_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    summary = audit_cambridge_parity(
        native_dir=Path(args.native_dir),
        stdloc_dir=Path(args.stdloc_dir),
        output_dir=Path(args.output_dir),
        scene=args.scene,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

