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
from loc_gs.teacher.render_teacher_jobs import build_render_teacher_jobs, load_online_episode_rows


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
    online_episodes: str | Path,
    require_render_ready: bool,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal render teacher job manifest")
    return {
        "schema_version": "internal_render_teacher_job_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [str(part) for part in command],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "training_teacher_job_generation",
        "dense_teacher_enabled": True,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "online_episodes": str(online_episodes),
        "hyperparameters": {"require_render_ready": bool(require_render_ready)},
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build training-only sparse-dense teacher jobs from render-ready episodes.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--online_episodes", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--require_render_ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="internal render teacher jobs")
    episodes = load_online_episode_rows(args.online_episodes)
    summary, jobs = build_render_teacher_jobs(
        episodes,
        scene=str(args.scene),
        split_name=split,
        require_render_ready=bool(args.require_render_ready),
    )
    command = [sys.executable, "-m", "loc_gs.scripts.build_internal_render_teacher_jobs", *(argv or sys.argv[1:])]
    manifest = build_manifest(
        scene=str(args.scene),
        split_name=split,
        command=command,
        online_episodes=args.online_episodes,
        require_render_ready=bool(args.require_render_ready),
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "teacher_jobs.jsonl").open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, sort_keys=True) + "\n")
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
