#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from loc_gs.diagnostics.solver_consensus_support import build_solver_consensus_support


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
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
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command(argv: list[str] | None) -> str:
    values = sys.argv if argv is None else [sys.argv[0], *argv]
    return " ".join(shlex.quote(str(item)) for item in values)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact per-Gaussian solver-consensus support from audited feedback_bank_v2."
    )
    parser.add_argument("--feedback_bank", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gaussians", type=int, default=None)
    parser.add_argument("--support_threshold", type=float, default=0.5)
    parser.add_argument("--reprojection_quality_threshold_px", type=float, default=8.0)
    parser.add_argument("--hard_negative_descriptor_score_min", type=float, default=0.5)
    parser.add_argument("--hard_negative_reprojection_error_px_min", type=float, default=8.0)
    parser.add_argument("--hard_negative_penalty", type=float, default=0.75)
    parser.add_argument("--dense_worsen_penalty", type=float, default=0.60)
    parser.add_argument("--dense_delta_bad_cm", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    artifact = build_solver_consensus_support(
        args.feedback_bank,
        num_gaussians=args.num_gaussians,
        support_threshold=args.support_threshold,
        reprojection_quality_threshold_px=args.reprojection_quality_threshold_px,
        hard_negative_descriptor_score_min=args.hard_negative_descriptor_score_min,
        hard_negative_reprojection_error_px_min=args.hard_negative_reprojection_error_px_min,
        hard_negative_penalty=args.hard_negative_penalty,
        dense_worsen_penalty=args.dense_worsen_penalty,
        dense_delta_bad_cm=args.dense_delta_bad_cm,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "solver_consensus_support.pt"
    manifest_path = output_dir / "manifest.json"
    torch.save(artifact, artifact_path)

    metadata = dict(artifact["metadata"])
    manifest = {
        "schema": "solver_consensus_support_manifest_v1",
        "artifact": artifact_path.name,
        "artifact_path": str(artifact_path),
        "feedback_bank": str(args.feedback_bank),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command(argv),
        "scene": str(metadata.get("scene", "")),
        "split": str(metadata.get("split", "")),
        "checkpoint_path": "",
        "map_path": "",
        "data_roots": {},
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": False,
        "rho_feedback_enabled": False,
        "hyperparameters": dict(metadata.get("hyperparameters", {})),
        "metadata": metadata,
    }
    _write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "manifest_path": str(manifest_path),
                "selected_count": int(metadata.get("selected_count", 0)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
