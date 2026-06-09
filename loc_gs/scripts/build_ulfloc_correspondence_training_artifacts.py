#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from loc_gs.feedback.ulfloc_correspondence_training_artifacts import (
    build_ulfloc_correspondence_training_artifacts,
)


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception as exc:
        return f"unknown: {exc}"


def _git_status(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(root), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ULF-Loc detector, match scorer, PnP ranking, and conflict targets from solver-feedback correspondences."
    )
    parser.add_argument("--correspondence_supervision", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--sigma_px", default=1.0, type=float)
    parser.add_argument("--hard_negative_max_per_query", default=64, type=int)
    parser.add_argument("--hard_negative_min_positive_distance_px", default=16.0, type=float)
    parser.add_argument("--hard_negative_grid_size", default=8, type=int)
    parser.add_argument("--hard_negative_max_per_cell", default=2, type=int)
    parser.add_argument("--solver_validity_power", default=0.0, type=float)
    parser.add_argument(
        "--detector_residual_alpha",
        default=0.0,
        type=float,
        help="Clamp solver-feedback detector residual to geomw teacher. Values above 0.1 are clipped.",
    )
    parser.add_argument(
        "--materialize_detector_heatmaps",
        action="store_true",
        help="Store full per-query heatmaps. Default stores compact point targets and rasterizes during training.",
    )
    return parser


def _compact_detector_targets(targets: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    compact: dict[str, dict[str, Any]] = {}
    for query_id, entry in sorted(targets.items()):
        compact[str(query_id)] = {
            "gaussian_ids": entry["gaussian_ids"],
            "keypoint_yx": entry["keypoint_yx"],
            "support_weights": entry["support_weights"],
            "teacher_support_weights": entry.get("teacher_support_weights", entry["support_weights"]),
            "solver_validity_weights": entry.get(
                "solver_validity_weights",
                torch.ones_like(entry["support_weights"]),
            ),
            "negative_gaussian_ids": entry.get("negative_gaussian_ids", torch.empty(0, dtype=torch.long)),
            "negative_keypoint_yx": entry.get("negative_keypoint_yx", torch.empty(0, 2, dtype=torch.float32)),
            "negative_weights": entry.get("negative_weights", torch.empty(0, dtype=torch.float32)),
            "positive_count": int(entry.get("positive_count", 0)),
            "negative_count": int(entry.get("negative_count", 0)),
            "target_metadata": dict(entry.get("target_metadata", {})),
        }
    return compact


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    supervision_path = Path(args.correspondence_supervision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supervision = _load_json(supervision_path)
    artifact = build_ulfloc_correspondence_training_artifacts(
        supervision,
        height=int(args.height),
        width=int(args.width),
        sigma_px=float(args.sigma_px),
        hard_negative_max_per_query=int(args.hard_negative_max_per_query),
        hard_negative_min_positive_distance_px=float(args.hard_negative_min_positive_distance_px),
        hard_negative_grid_size=int(args.hard_negative_grid_size),
        hard_negative_max_per_cell=int(args.hard_negative_max_per_cell),
        solver_validity_power=float(args.solver_validity_power),
        detector_residual_alpha=float(args.detector_residual_alpha),
    )

    detector_targets = (
        artifact["detector_targets"]
        if bool(args.materialize_detector_heatmaps)
        else _compact_detector_targets(artifact["detector_targets"])
    )
    detector_metadata = dict(artifact["metadata"])
    detector_metadata["detector_target_storage"] = "heatmap" if bool(args.materialize_detector_heatmaps) else "points"
    detector_payload = {
        "schema_version": "ulfloc_detector_targets_from_solver_feedback_v1",
        "split_name": artifact["split_name"],
        "targets": detector_targets,
        "metadata": detector_metadata,
        "split_audit": artifact["split_audit"],
    }
    torch.save(detector_payload, output_dir / "detector_targets.pt")
    torch.save(artifact["match_scorer"], output_dir / "match_scorer.pt")
    _write_json(output_dir / "pnp_ranking.json", artifact["pnp_ranking"])
    _write_json(output_dir / "descriptor_conflict_graph.json", artifact["conflict_graph"])
    _write_json(output_dir / "metrics_summary.json", detector_metadata)
    _write_json(output_dir / "split_audit.json", artifact["split_audit"])

    manifest = {
        "schema_version": "ulfloc_correspondence_training_artifacts_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "command": sys.argv,
        "correspondence_supervision": str(supervision_path),
        "output_dir": str(output_dir),
        "height": int(args.height),
        "width": int(args.width),
        "sigma_px": float(args.sigma_px),
        "hard_negative_max_per_query": int(args.hard_negative_max_per_query),
        "hard_negative_min_positive_distance_px": float(args.hard_negative_min_positive_distance_px),
        "hard_negative_grid_size": int(args.hard_negative_grid_size),
        "hard_negative_max_per_cell": int(args.hard_negative_max_per_cell),
        "solver_validity_power": float(args.solver_validity_power),
        "detector_residual_alpha": float(args.detector_residual_alpha),
        "paper_safe_role": "train_selfmap_solver_feedback_training_targets",
        "metrics": detector_metadata,
        "split_audit": artifact["split_audit"],
        "outputs": {
            "detector_targets": str(output_dir / "detector_targets.pt"),
            "match_scorer": str(output_dir / "match_scorer.pt"),
            "pnp_ranking": str(output_dir / "pnp_ranking.json"),
            "descriptor_conflict_graph": str(output_dir / "descriptor_conflict_graph.json"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "command.txt").write_text(_command() + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(repo_root), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **detector_metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
