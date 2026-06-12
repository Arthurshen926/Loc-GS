#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.simulation.render_manifest import build_render_manifest_from_plan_rows, load_simulation_plan_rows
from loc_gs.sparse.audit import reject_test_split


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    simulation_plan: str | Path,
    render_root: str | Path,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal 3DGS render manifest manifest")
    return {
        "schema_version": "internal_render_manifest_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "render_manifest_audit",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "simulation_plan": str(simulation_plan),
        "render_root": str(render_root),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an audited manifest for expected internal 3DGS render assets.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--simulation_plan", type=Path, required=True)
    parser.add_argument("--render_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--require_rgb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal 3DGS render manifest")
    rows = load_simulation_plan_rows(args.simulation_plan)
    summary, records = build_render_manifest_from_plan_rows(
        rows,
        scene=str(args.scene),
        split_name=split,
        render_root=args.render_root,
    )
    if bool(args.require_rgb) and int(summary["missing_rgb_count"]) > 0:
        raise ValueError(f"missing RGB render assets: {summary['missing_rgb_count']}")
    command = [sys.executable, "-m", "loc_gs.scripts.build_internal_render_manifest", *(argv or sys.argv[1:])]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        simulation_plan=args.simulation_plan,
        render_root=args.render_root,
    )
    missing_render_ids = [str(record["synthetic_query_id"]) for record in records if not bool(record["rgb_exists"])]
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "render_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (args.output_dir / "missing_render_ids.txt").write_text(
        "\n".join(missing_render_ids) + ("\n" if missing_render_ids else ""),
        encoding="utf-8",
    )
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
