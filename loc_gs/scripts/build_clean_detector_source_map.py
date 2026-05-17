from __future__ import annotations

import argparse
import json

from loc_gs.stdloc_native.clean_source_map import build_clean_detector_source_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean STDLoc source map by replacing detector/ with a rebuilt detector payload."
    )
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--rebuilt_detector_dir", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--no_overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_clean_detector_source_map(
        source_map=args.source_map,
        rebuilt_detector_dir=args.rebuilt_detector_dir,
        output_map=args.output_map,
        scene=args.scene,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
