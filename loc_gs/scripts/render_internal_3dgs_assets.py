#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.simulation.render_runner import (
    build_feature_map_cache_from_render_records,
    load_render_manifest_records,
    render_assets_from_manifest_records,
    render_record_with_internal_3dgs,
)
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
    render_manifest: str | Path,
    gaussian_ply: str | Path | None,
    feature_checkpoint: str | Path | None,
    feature_map_cache: str | Path,
    render_feature_map: bool,
    dry_run: bool,
    max_records: int | None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal 3DGS render runner manifest")
    return {
        "schema_version": "internal_render_runner_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "online_3dgs_render_asset_generation",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "render_manifest": str(render_manifest),
        "gaussian_ply": None if gaussian_ply is None else str(gaussian_ply),
        "feature_checkpoint": None if feature_checkpoint is None else str(feature_checkpoint),
        "feature_map_cache": str(feature_map_cache),
        "hyperparameters": {
            "dry_run": bool(dry_run),
            "max_records": None if max_records is None else int(max_records),
            "render_feature_map": bool(render_feature_map),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render internal 3DGS assets from an audited render manifest.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--render_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gaussian_ply", type=Path, default=None)
    parser.add_argument("--feature_checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--feature_map_cache_name", default="render_feature_map_cache.pt")
    parser.add_argument("--disable_feature_map", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal 3DGS render runner")
    if not bool(args.dry_run) and args.gaussian_ply is None:
        raise ValueError("--gaussian_ply is required unless --dry_run is set")
    records = load_render_manifest_records(args.render_manifest)
    if bool(args.dry_run):
        render_fn = lambda _record: {}
    else:
        render_fn = functools.partial(
            render_record_with_internal_3dgs,
            gaussian_ply=args.gaussian_ply,
            feature_checkpoint=args.feature_checkpoint,
            device=str(args.device),
            latent_dim=int(args.latent_dim),
            render_feature_map=not bool(args.disable_feature_map),
        )
    summary, updated = render_assets_from_manifest_records(
        records,
        scene=str(args.scene),
        split_name=split,
        render_fn=render_fn,
        dry_run=bool(args.dry_run),
        max_records=args.max_records,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_map_cache = args.output_dir / str(args.feature_map_cache_name)
    feature_summary = build_feature_map_cache_from_render_records(
        updated,
        output_cache=feature_map_cache,
        scene=str(args.scene),
        split_name=split,
    )
    summary = {
        **summary,
        "feature_map_cache": str(feature_map_cache),
        "feature_map_cache_entry_count": int(feature_summary["entry_count"]),
        "feature_map_cache_missing_count": int(feature_summary["missing_feature_map_count"]),
    }
    command = [sys.executable, "-m", "loc_gs.scripts.render_internal_3dgs_assets", *(argv or sys.argv[1:])]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        render_manifest=args.render_manifest,
        gaussian_ply=args.gaussian_ply,
        feature_checkpoint=args.feature_checkpoint,
        feature_map_cache=feature_map_cache,
        render_feature_map=not bool(args.disable_feature_map),
        dry_run=bool(args.dry_run),
        max_records=args.max_records,
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    with (args.output_dir / "render_manifest.updated.jsonl").open("w", encoding="utf-8") as handle:
        for record in updated:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
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
