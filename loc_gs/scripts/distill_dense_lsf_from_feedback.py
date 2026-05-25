#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.dense_support.distill_dense_lsf import distill_dense_lsf_targets


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill dense LSF support targets from a feedback_bank_v2.")
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--num_gaussians", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--keep_threshold", type=float, default=0.5)
    parser.add_argument("--delta_scale_cm", type=float, default=10.0)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--output_json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    payload = distill_dense_lsf_targets(
        args.feedback_bank,
        num_gaussians=int(args.num_gaussians) if int(args.num_gaussians) > 0 else None,
        smoothing=float(args.smoothing),
        keep_threshold=float(args.keep_threshold),
        delta_scale_cm=float(args.delta_scale_cm),
    )
    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_pt)
    metadata = dict(payload["metadata"])
    summary = {
        **metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "output_pt": str(output_pt),
        "target_mean_observed": float(payload["dense_lsf_target"][payload["observed_count"] > 0].mean().item())
        if bool((payload["observed_count"] > 0).any())
        else 0.0,
        "confidence_mean_observed": float(payload["dense_lsf_confidence"][payload["observed_count"] > 0].mean().item())
        if bool((payload["observed_count"] > 0).any())
        else 0.0,
    }
    _write_json(args.output_json, summary)
    print(json.dumps({"output_pt": str(output_pt), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
