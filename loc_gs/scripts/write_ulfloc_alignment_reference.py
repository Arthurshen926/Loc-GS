#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.reporting.ulfloc_alignment import (
    LOCAL_REPRODUCTION_REQUIRED,
    PAPER_REPORTED,
    build_ulfloc_alignment_reference,
    write_ulfloc_alignment_reference,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a reporting-only ULF-Loc reproduction alignment reference."
    )
    parser.add_argument("--checkout-path", default="", help="Optional local ULF-Loc checkout path.")
    parser.add_argument("--result-root", default="", help="Optional local ULF-Loc Cambridge output root.")
    parser.add_argument("--official-head", default="", help="Official ULF-Loc git HEAD used for the reference.")
    parser.add_argument("--code-reproduced", action="store_true", help="Mark reproduced only if outputs also exist.")
    parser.add_argument(
        "--reference-mode",
        choices=(LOCAL_REPRODUCTION_REQUIRED, PAPER_REPORTED),
        default=LOCAL_REPRODUCTION_REQUIRED,
        help="Use paper_reported when ULF-Loc is included only as an external paper baseline.",
    )
    parser.add_argument(
        "--paper-reported-ok",
        action="store_true",
        help="Alias for --reference-mode paper_reported.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    reference_mode = PAPER_REPORTED if args.paper_reported_ok else args.reference_mode
    payload = build_ulfloc_alignment_reference(
        checkout_path=Path(args.checkout_path) if args.checkout_path else None,
        result_root=Path(args.result_root) if args.result_root else None,
        official_head=args.official_head,
        code_reproduced=bool(args.code_reproduced),
        reference_mode=reference_mode,
        notes=args.notes,
    )
    write_ulfloc_alignment_reference(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_json": args.output_json,
                "code_reproduced": payload["method"]["code_reproduced"],
                "reference_mode": payload["method"]["reference_mode"],
                "reporting_allowed": payload["method"]["reporting_allowed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
