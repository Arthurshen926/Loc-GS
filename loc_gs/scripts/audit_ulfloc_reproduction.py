#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.reporting.ulfloc_reproduction_audit import (
    audit_markdown,
    build_ulfloc_reproduction_audit,
)


def _git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(cwd), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local ULF-Loc reproduction readiness.")
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--result_root", default=None, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    report = build_ulfloc_reproduction_audit(ulf_root=args.ulf_root, result_root=args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(audit_markdown(report), encoding="utf-8")
    manifest = {
        "schema_version": "ulfloc_reproduction_audit_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "ulf_root": str(args.ulf_root),
        "result_root": str(args.result_root or Path(args.ulf_root) / "outputs"),
        "output_dir": str(output_dir),
        "gap_count": len(report.get("gaps", [])),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path.cwd()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

