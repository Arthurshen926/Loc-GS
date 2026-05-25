from __future__ import annotations

import argparse
import json
from pathlib import Path

from loc_gs.reporting.frozen_profile_coverage import build_frozen_profile_coverage


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check q-profile runtime/memory coverage for a frozen scene-level LSF recipe."
    )
    parser.add_argument("--frozen-recipe", required=True, help="Path to frozen recipe manifest JSON.")
    parser.add_argument("--profile-roots", nargs="+", required=True, help="Profile output roots or run directories.")
    parser.add_argument("--required-queries", type=int, default=5)
    parser.add_argument("--output-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    frozen_path = Path(args.frozen_recipe)
    frozen_manifest = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not isinstance(frozen_manifest, dict):
        raise ValueError(f"expected JSON object in {frozen_path}")
    report = build_frozen_profile_coverage(
        frozen_manifest,
        profile_roots=[Path(root) for root in args.profile_roots],
        required_queries=args.required_queries,
        frozen_recipe_path=str(frozen_path),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
