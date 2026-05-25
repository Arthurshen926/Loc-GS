#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loc_gs.reporting.frozen_recipe import build_frozen_recipe_manifest


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _parse_report_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"acceptance report spec must be Label=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"missing label in acceptance report spec {spec!r}")
    return label, Path(path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a frozen scene-level recipe manifest from fixed-recipe acceptance reports."
    )
    parser.add_argument("--acceptance-report", action="append", default=[], help="Label=acceptance_report.json")
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    reports = {}
    for spec in args.acceptance_report:
        label, path = _parse_report_spec(spec)
        reports[label] = _load_json(path)
    payload = build_frozen_recipe_manifest(
        reports,
        freeze_id=args.freeze_id,
        notes=args.notes,
        require_passed=not args.allow_failed_gate,
    )
    payload["command"] = " ".join(sys.argv if argv is None else ["write_frozen_recipe_manifest", *argv])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "paper_facing_ready": payload["paper_facing_ready"],
                "acceptance_gates_passed": payload["checks"]["acceptance_gates_passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
