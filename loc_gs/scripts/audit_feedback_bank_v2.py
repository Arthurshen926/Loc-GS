#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.feedback.audit import audit_feedback_bank_v2


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a Loc-GS feedback bank for v2 query-level supervision.")
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--fail_on_error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    payload = audit_feedback_bank_v2(args.feedback_bank)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.fail_on_error and payload["audit_status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
