#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from loc_gs.eval.locgs_metrics import summarize_lsf_eval
from loc_gs.stdloc_native.results import load_stdloc_query_results


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a STDLoc/Loc-GS run with the LSF evaluation protocol.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--landmark_count", type=int, default=None)
    parser.add_argument("--map_size_mb", type=float, default=None)
    parser.add_argument("--base_map_s", type=float, default=None)
    parser.add_argument("--feedback_cache_s", type=float, default=None)
    parser.add_argument("--selector_train_s", type=float, default=None)
    parser.add_argument("--export_s", type=float, default=None)
    parser.add_argument("--peak_gpu_mb", type=float, default=None)
    parser.add_argument("--output_json", required=True)
    return parser


def _offline_costs_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    keys = ("base_map_s", "feedback_cache_s", "selector_train_s", "export_s", "peak_gpu_mb")
    payload = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
    return payload or None


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run_dir = Path(args.run_dir)
    rows = load_stdloc_query_results(run_dir)
    timing = _load_json_if_exists(run_dir / "timing_profile.json")
    metrics = _load_json_if_exists(run_dir / "metrics_summary.json") or {}
    scene = args.scene or str(metrics.get("scene") or run_dir.name)
    method = args.method or str(metrics.get("method") or run_dir.name)
    landmark_count = args.landmark_count
    if landmark_count is None and "landmark_count" in metrics:
        landmark_count = int(metrics["landmark_count"])
    report = summarize_lsf_eval(
        rows,
        scene=scene,
        method=method,
        landmark_count=landmark_count,
        timing_profile=timing,
        offline_costs=_offline_costs_from_args(args),
        map_size_mb=args.map_size_mb,
    )
    report.update(
        {
            "run_dir": str(run_dir),
            "git_commit": _git_commit(),
            "protocol": "lsf_eval_v1",
            "notes": {
                "official_metrics_unmodified": True,
                "initial_pose_stage": "sparse",
                "final_pose_stage": "dense",
            },
        }
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_json": str(output)}, indent=2))


if __name__ == "__main__":
    main()

