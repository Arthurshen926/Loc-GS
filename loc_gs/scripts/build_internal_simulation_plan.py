#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loc_gs.core.camera import load_camera_records
from loc_gs.simulation.query_sampler import (
    SimulationSamplerConfig,
    sample_simulated_queries,
    summarize_simulation_plan,
)
from loc_gs.sparse.audit import reject_test_split


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an audited internal 3DGS simulation query plan.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--cameras_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--translation_std_m", type=float, default=0.25)
    parser.add_argument("--yaw_std_deg", type=float, default=5.0)
    parser.add_argument("--pitch_std_deg", type=float, default=2.0)
    parser.add_argument("--roll_std_deg", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    split = reject_test_split(str(args.split_name), purpose="3DGS simulation plan CLI")
    cfg = SimulationSamplerConfig(
        sample_count=int(args.sample_count),
        seed=int(args.seed),
        translation_std_m=float(args.translation_std_m),
        yaw_std_deg=float(args.yaw_std_deg),
        pitch_std_deg=float(args.pitch_std_deg),
        roll_std_deg=float(args.roll_std_deg),
    )
    specs = sample_simulated_queries(load_camera_records(args.cameras_json), scene=str(args.scene), split_name=split, cfg=cfg)
    summary = summarize_simulation_plan(specs)
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
        "paper_safe_for_tuning": True,
    }
    manifest = {
        "schema_version": "internal_3dgs_simulation_plan_manifest_v1",
        "method": "sparse_dense_distilled_internal",
        "scene": str(args.scene),
        "split_name": split,
        "git_commit": _git_commit(),
        "command": [sys.executable, "-m", "loc_gs.scripts.build_internal_simulation_plan", *(argv or sys.argv[1:])],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inference_stage": "simulation_plan_generation",
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
        "cameras_json": str(args.cameras_json),
        "hyperparameters": {
            "sample_count": int(args.sample_count),
            "seed": int(args.seed),
            "translation_std_m": float(args.translation_std_m),
            "yaw_std_deg": float(args.yaw_std_deg),
            "pitch_std_deg": float(args.pitch_std_deg),
            "roll_std_deg": float(args.roll_std_deg),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "simulation_plan.jsonl").open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec.to_json_dict(), sort_keys=True) + "\n")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
