#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.sparse.audit import reject_test_split


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_sparse_distilled_manifest(
    *,
    scene: str,
    split_name: str,
    command: Sequence[str],
    dense_teacher_enabled: bool = False,
    data_root: str | Path | None = None,
    map_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="sparse-distilled evaluation manifest")
    return {
        "schema_version": "sparse_dense_distilled_internal_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "sparse_only",
        "dense_teacher_enabled": bool(dense_teacher_enabled),
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "data_root": None if data_root is None else str(data_root),
        "map_path": None if map_path is None else str(map_path),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the internal sparse-dense distilled sparse-only mainline.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", default="train_dev")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--map_path", type=Path, default=None)
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--dense_teacher_enabled", action="store_true")
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    command = [sys.executable, "-m", "loc_gs.scripts.eval_sparse_distilled_cambridge", *(argv or sys.argv[1:])]
    manifest = build_sparse_distilled_manifest(
        scene=str(args.scene),
        split_name=str(args.split_name),
        command=command,
        dense_teacher_enabled=bool(args.dense_teacher_enabled),
        data_root=args.data_root,
        map_path=args.map_path,
        checkpoint_path=args.checkpoint_path,
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if bool(args.dry_run):
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    raise NotImplementedError("non-dry-run sparse distilled evaluation will be added after internal sparse parity")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
