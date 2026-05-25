#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from loc_gs.diagnostics.localization_support_field import write_localization_support_field


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a canonical LSF-Loc localization support field artifact.")
    parser.add_argument("--solver_consensus_support_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--solver_admissibility_path", default="")
    parser.add_argument("--support_score_mode", choices=("raw", "rank_observed"), default="raw")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    manifest = write_localization_support_field(
        solver_consensus_support_path=args.solver_consensus_support_path,
        solver_admissibility_path=args.solver_admissibility_path or None,
        output_dir=args.output_dir,
        support_score_mode=args.support_score_mode,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
