#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.multihypothesis_pnp import build_multihypothesis_diagnostic_plan


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v12 multi-hypothesis PnP diagnostic artifact.")
    parser.add_argument("--matches_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_hypotheses", type=int, default=8)
    parser.add_argument("--scene", default="unknown")
    parser.add_argument("--split_name", default="unknown")
    parser.add_argument("--diagnostic", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    matches: Any = json.loads(Path(args.matches_json).read_text(encoding="utf-8"))
    if not isinstance(matches, list):
        raise TypeError("matches_json must contain a list")
    result = build_multihypothesis_diagnostic_plan(
        matches=matches,
        diagnostic=bool(args.diagnostic),
        max_hypotheses=int(args.max_hypotheses),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "multihypothesis_pnp_diagnostic.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "enabled": True,
        "scene": str(args.scene),
        "split_name": str(args.split_name).strip() or "unknown",
        "recipe": "lsf_v12_multihyp_diagnostic",
        "artifact": str(artifact_path),
        "single_path_deployment": False,
        "branch_selection": True,
        "diagnostic_only": True,
        "main_method_allowed": False,
    }
    metrics = {
        "hypothesis_count": int(result["metadata"]["hypothesis_count"]),
        "input_match_count": int(result["metadata"]["input_match_count"]),
        "diagnostic_only": True,
    }
    manifest = {
        "method": "loc_gs_lsf_v12_multihypothesis_pnp_diagnostic",
        **metadata,
        "matches_json": str(args.matches_json),
        "max_hypotheses": int(args.max_hypotheses),
    }
    split_audit = artifact_split_audit(metadata, branch_selection=True)
    write_artifact_audit_bundle(
        output_dir,
        manifest=manifest,
        command=_command(),
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    print(json.dumps({"artifact": str(artifact_path), **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
