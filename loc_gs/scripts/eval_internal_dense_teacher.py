#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loc_gs.scripts.eval_sparse_conditioned_dense_control import (
    _macro_summary,
    _read_case_rows,
    _write_json,
    evaluate_scene,
)
from loc_gs.scripts.locgsctl import DEFAULT_DATA_ROOT, DEFAULT_MAP_ROOT
from loc_gs.sparse.audit import reject_test_split
from loc_gs.stdloc_native.commands import CAMBRIDGE_SCENES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENSE_CFG = REPO_ROOT / "third_party/stdloc/configs/stdloc_spgs_cambridge_dense1.yaml"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _sampled_count(map_path: Path) -> int | None:
    sampled_idx = map_path / "detector" / "sampled_idx.pkl"
    if not sampled_idx.exists():
        return None
    try:
        with sampled_idx.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    try:
        return int(len(payload))
    except TypeError:
        return int(payload.numel()) if hasattr(payload, "numel") else None


def create_dense_teacher_context(
    *,
    context_root: Path,
    scene: str,
    split_name: str,
    map_root: Path,
    data_root: Path,
    cfg_path: Path,
    command: Sequence[str],
) -> Path:
    split = reject_test_split(str(split_name), purpose="internal dense teacher context")
    scene_name = str(scene)
    run_dir = Path(context_root) / scene_name
    run_dir.mkdir(parents=True, exist_ok=True)
    map_path = Path(map_root) / scene_name
    scene_root = Path(data_root) / scene_name
    sampled_count = _sampled_count(map_path)
    manifest = {
        "schema_version": "internal_dense_teacher_context_v1",
        "method": "internal_sparse_dense_teacher",
        "scene": scene_name,
        "split_name": split,
        "git_commit": _git_commit(),
        "command": list(command),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_roots": [str(scene_root)],
        "map_path": str(map_path),
        "sampled_count": sampled_count,
        "native_sampled_count_expected": 16384,
        "native_sampled_count_status": "passed" if sampled_count == 16384 else "mismatch",
        "dense_teacher_enabled": True,
        "external_runtime_dependency": "forbidden",
        "hyperparameters": {
            "cfg": str(cfg_path),
            "sparse_landmark_path": "detector/sampled_idx.pkl",
            "sparse_detector_path": "detector/30000_detector.pth",
            "dense_stage": "stdloc_style_render_refinement",
        },
    }
    split_audit = {
        "schema_version": "internal_dense_teacher_split_audit_v1",
        "scene": scene_name,
        "split_name": split,
        "official_test_used": False,
        "official_test_eval": False,
        "test_split_used": False,
        "paper_safe_for_tuning": True,
        "role": "train_dev_teacher_upper_bound",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate internal sparse-dense teacher dense stage.")
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--context_root", type=Path, default=None)
    parser.add_argument("--map_root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_DENSE_CFG)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=1)
    parser.add_argument("--progress_interval", type=int, default=10)
    parser.add_argument("--case_csv", action="append", default=[])
    parser.add_argument("--case_type", action="append", default=[])
    parser.add_argument("--phase0_split", choices=["train", "val", "all"], default="all")
    parser.add_argument(
        "--candidate_render_control",
        choices=["none", "sparse_conditioned"],
        default="none",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal dense teacher eval")
    output_dir = Path(args.output_dir)
    context_root = Path(args.context_root) if args.context_root else output_dir / "_context"
    case_rows = (
        _read_case_rows(list(args.case_csv), case_types=list(args.case_type), phase0_split=str(args.phase0_split))
        if args.case_csv
        else {}
    )
    scenes = tuple(args.scene) if args.scene else tuple(sorted(case_rows.keys())) if case_rows else CAMBRIDGE_SCENES
    for scene in scenes:
        create_dense_teacher_context(
            context_root=context_root,
            scene=str(scene),
            split_name=split,
            map_root=Path(args.map_root),
            data_root=Path(args.data_root),
            cfg_path=Path(args.cfg),
            command=[sys.executable, "-m", "loc_gs.scripts.eval_internal_dense_teacher", *(argv or sys.argv[1:])],
        )
    summaries = [
        evaluate_scene(
            candidate_root=context_root,
            output_dir=output_dir,
            scene=str(scene),
            eval_split=str(args.eval_split),
            max_queries=int(args.max_queries),
            query_stride=int(args.query_stride),
            progress_interval=int(args.progress_interval),
            candidate_render_control=str(args.candidate_render_control),
            case_rows_by_image=case_rows.get(str(scene), {}),
        )
        for scene in scenes
    ]
    macro = _macro_summary(summaries)
    _write_json(output_dir / "summary.json", {"scenes": summaries, "macro": macro})
    print(json.dumps({"output_dir": str(output_dir), "macro": macro}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

