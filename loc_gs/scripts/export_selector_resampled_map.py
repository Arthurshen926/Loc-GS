#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData

from loc_gs.stdloc_native.selector_resampling import (
    hard_negative_risk_from_episode_cache,
    hard_query_support_from_episode_cache,
    native_support_reservation_from_episode_cache,
    positive_support_from_episode_cache,
    pose_information_from_episode_cache,
    query_coverage_reservation_from_episode_cache,
    resample_detector_landmarks,
    selector_from_checkpoint,
    support_preservation_audit_from_episode_cache,
    write_resampled_detector_payload,
)
from loc_gs.stdloc_native.unified_lff_export import build_unified_lff_map
from loc_gs.stdloc_native.lff_export import _latest_point_cloud_path


def _load_xyz(path: Path) -> torch.Tensor:
    vertex = PlyData.read(str(path))["vertex"].data
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    return torch.as_tensor(xyz, dtype=torch.float32)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a selector-guided resampled STDLoc map.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--descriptor_mode", choices=("native", "checkpoint"), default="native")
    parser.add_argument("--budget", default="same_as_source")
    parser.add_argument("--selector_weight", type=float, default=1.0)
    parser.add_argument("--source_score_weight", type=float, default=1.0)
    parser.add_argument("--selector_transform", choices=("identity", "rank", "minmax"), default="identity")
    parser.add_argument("--coverage_grid", type=int, default=0)
    parser.add_argument("--candidate_pool", choices=("all_gaussians", "source_sampled"), default="source_sampled")
    parser.add_argument("--preserve_source_order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_retention_fraction", type=float, default=0.0)
    parser.add_argument("--hard_negative_cache", default="")
    parser.add_argument("--hard_negative_weight", type=float, default=0.0)
    parser.add_argument("--hard_negative_score_threshold", type=float, default=0.0)
    parser.add_argument("--hard_negative_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--positive_support_cache", default="")
    parser.add_argument("--positive_support_weight", type=float, default=0.0)
    parser.add_argument("--positive_support_score_threshold", type=float, default=0.0)
    parser.add_argument("--positive_support_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--pose_information_cache", default="")
    parser.add_argument("--pose_information_weight", type=float, default=0.0)
    parser.add_argument("--pose_information_score_threshold", type=float, default=0.0)
    parser.add_argument("--pose_information_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--hard_query_support_cache", default="")
    parser.add_argument("--hard_query_support_weight", type=float, default=0.0)
    parser.add_argument("--hard_query_support_score_threshold", type=float, default=0.0)
    parser.add_argument("--hard_query_support_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--support_guard_cache", default="")
    parser.add_argument("--support_guard_fraction", type=float, default=0.0)
    parser.add_argument("--support_guard_score_threshold", type=float, default=0.0)
    parser.add_argument("--support_guard_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--support_guard_margin_threshold", type=float, default=None)
    parser.add_argument("--support_guard_hard_query_fraction", type=float, default=0.25)
    parser.add_argument("--support_guard_preserve_counts", action="store_true")
    parser.add_argument("--query_coverage_cache", default="")
    parser.add_argument("--query_coverage_fraction", type=float, default=0.0)
    parser.add_argument("--query_coverage_score_threshold", type=float, default=0.0)
    parser.add_argument("--query_coverage_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--query_coverage_margin_threshold", type=float, default=None)
    parser.add_argument("--query_coverage_hard_query_fraction", type=float, default=0.25)
    parser.add_argument("--strict_support_cache", default="")
    parser.add_argument("--strict_support_fraction", type=float, default=0.0)
    parser.add_argument("--strict_support_score_threshold", type=float, default=0.0)
    parser.add_argument("--strict_support_reprojection_threshold_px", type=float, default=None)
    parser.add_argument("--protected_source_map", default="")
    parser.add_argument("--protected_source_idx_path", default="")
    parser.add_argument("--protected_source_fraction", type=float, default=None)
    parser.add_argument("--locability_blend", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_overwrite", action="store_true")
    return parser


def _budget_value(value: str) -> str | int:
    return "same_as_source" if str(value) == "same_as_source" else int(value)


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


def _command_from_args(args: argparse.Namespace) -> str:
    parts = ["python", "-m", "loc_gs.scripts.export_selector_resampled_map"]
    for key, value in sorted(vars(args).items()):
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                parts.append(option)
            continue
        if value is None:
            continue
        parts.extend([option, str(value)])
    return " ".join(shlex.quote(part) for part in parts)


def _feedback_split_from_payload(payload: dict[str, Any]) -> str:
    for key in (
        "query_coverage_metadata",
        "support_guard_metadata",
        "hard_query_support_metadata",
        "pose_information_metadata",
        "positive_support_metadata",
        "hard_negative_metadata",
        "strict_support_metadata",
    ):
        metadata = payload.get(key)
        if not isinstance(metadata, dict):
            continue
        split = metadata.get("feedback_bank_split_name") or metadata.get("split_name")
        if split:
            return str(split)
        split_audit = metadata.get("split_audit")
        if isinstance(split_audit, dict):
            split = split_audit.get("feedback_bank_split_name") or split_audit.get("split_name")
            if split:
                return str(split)
    return "unknown"


def _resampling_manifest(
    args: argparse.Namespace,
    payload: dict[str, Any],
    source_ply: Path,
    *,
    command: str,
) -> dict[str, Any]:
    hyperparameters = {
        "budget": str(args.budget),
        "selector_weight": float(args.selector_weight),
        "source_score_weight": float(args.source_score_weight),
        "selector_transform": str(args.selector_transform),
        "coverage_grid": int(args.coverage_grid),
        "candidate_pool": str(args.candidate_pool),
        "preserve_source_order": bool(args.preserve_source_order),
        "source_retention_fraction": float(args.source_retention_fraction),
        "hard_negative_weight": float(args.hard_negative_weight),
        "hard_negative_score_threshold": float(args.hard_negative_score_threshold),
        "hard_negative_reprojection_threshold_px": args.hard_negative_reprojection_threshold_px,
        "positive_support_weight": float(args.positive_support_weight),
        "positive_support_score_threshold": float(args.positive_support_score_threshold),
        "positive_support_reprojection_threshold_px": args.positive_support_reprojection_threshold_px,
        "pose_information_weight": float(args.pose_information_weight),
        "pose_information_score_threshold": float(args.pose_information_score_threshold),
        "pose_information_reprojection_threshold_px": args.pose_information_reprojection_threshold_px,
        "hard_query_support_weight": float(args.hard_query_support_weight),
        "hard_query_support_score_threshold": float(args.hard_query_support_score_threshold),
        "hard_query_support_reprojection_threshold_px": args.hard_query_support_reprojection_threshold_px,
        "support_guard_fraction": float(payload.get("support_guard_fraction", 0.0)),
        "support_guard_score_threshold": float(args.support_guard_score_threshold),
        "support_guard_reprojection_threshold_px": args.support_guard_reprojection_threshold_px,
        "support_guard_margin_threshold": args.support_guard_margin_threshold,
        "support_guard_hard_query_fraction": float(args.support_guard_hard_query_fraction),
        "support_guard_preserve_counts": bool(args.support_guard_preserve_counts),
        "query_coverage_fraction": float(payload.get("query_coverage_fraction", 0.0)),
        "query_coverage_score_threshold": float(args.query_coverage_score_threshold),
        "query_coverage_reprojection_threshold_px": args.query_coverage_reprojection_threshold_px,
        "query_coverage_margin_threshold": args.query_coverage_margin_threshold,
        "query_coverage_hard_query_fraction": float(args.query_coverage_hard_query_fraction),
        "strict_support_fraction": float(args.strict_support_fraction),
        "strict_support_score_threshold": float(args.strict_support_score_threshold),
        "strict_support_reprojection_threshold_px": args.strict_support_reprojection_threshold_px,
        "protected_source_fraction": float(payload.get("protected_source_fraction", 0.0)),
        "locability_blend": float(args.locability_blend),
        "seed": int(args.seed),
    }
    return {
        "method": "loc_gs_selector_guided_resampling",
        "git_commit": _git_commit(),
        "command": command,
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "scene": Path(args.source_map).name,
        "split": _feedback_split_from_payload(payload),
        "source_map": str(args.source_map),
        "checkpoint_path": str(args.checkpoint_path),
        "output_map": str(args.output_map),
        "map_path": str(args.output_map),
        "data_roots": {
            "source_map": str(args.source_map),
            "checkpoint_path": str(args.checkpoint_path),
            "output_map": str(args.output_map),
        },
        "hyperparameters": hyperparameters,
        "source_ply": str(source_ply),
        "descriptor_mode": str(args.descriptor_mode),
        "budget": str(args.budget),
        "source_sampled_count": int(payload["source_count"]),
        "output_sampled_count": int(payload["output_count"]),
        "sampled_idx_changed": bool(payload["sampled_idx_changed"]),
        "selector_weight": float(args.selector_weight),
        "source_score_weight": float(args.source_score_weight),
        "selector_transform": str(args.selector_transform),
        "coverage_grid": int(args.coverage_grid),
        "candidate_pool": str(args.candidate_pool),
        "preserve_source_order": bool(args.preserve_source_order),
        "source_retention_fraction": float(args.source_retention_fraction),
        "source_retained_count": int(payload.get("source_retained_count", 0)),
        "hard_negative_cache": str(args.hard_negative_cache),
        "hard_negative_weight": float(args.hard_negative_weight),
        "hard_negative_score_threshold": float(args.hard_negative_score_threshold),
        "hard_negative_reprojection_threshold_px": args.hard_negative_reprojection_threshold_px,
        "hard_negative_risk": dict(payload.get("hard_negative_metadata", {})),
        "positive_support_cache": str(args.positive_support_cache),
        "positive_support_weight": float(args.positive_support_weight),
        "positive_support_score_threshold": float(args.positive_support_score_threshold),
        "positive_support_reprojection_threshold_px": args.positive_support_reprojection_threshold_px,
        "positive_support": dict(payload.get("positive_support_metadata", {})),
        "pose_information_cache": str(args.pose_information_cache),
        "pose_information_weight": float(args.pose_information_weight),
        "pose_information_score_threshold": float(args.pose_information_score_threshold),
        "pose_information_reprojection_threshold_px": args.pose_information_reprojection_threshold_px,
        "pose_information": dict(payload.get("pose_information_metadata", {})),
        "hard_query_support_cache": str(args.hard_query_support_cache),
        "hard_query_support_weight": float(args.hard_query_support_weight),
        "hard_query_support_score_threshold": float(args.hard_query_support_score_threshold),
        "hard_query_support_reprojection_threshold_px": args.hard_query_support_reprojection_threshold_px,
        "hard_query_support": dict(payload.get("hard_query_support_metadata", {})),
        "support_guard_cache": str(args.support_guard_cache),
        "support_guard_fraction": float(payload.get("support_guard_fraction", 0.0)),
        "support_guard_score_threshold": float(args.support_guard_score_threshold),
        "support_guard_reprojection_threshold_px": args.support_guard_reprojection_threshold_px,
        "support_guard_margin_threshold": args.support_guard_margin_threshold,
        "support_guard_hard_query_fraction": float(args.support_guard_hard_query_fraction),
        "support_guard_preserve_counts": bool(args.support_guard_preserve_counts),
        "support_guard_reserved_count": int(payload.get("support_guard_reserved_count", 0)),
        "support_guard": dict(payload.get("support_guard_metadata", {})),
        "support_audit": dict(payload.get("support_audit", {})),
        "query_coverage_cache": str(args.query_coverage_cache),
        "query_coverage_fraction": float(payload.get("query_coverage_fraction", 0.0)),
        "query_coverage_score_threshold": float(args.query_coverage_score_threshold),
        "query_coverage_reprojection_threshold_px": args.query_coverage_reprojection_threshold_px,
        "query_coverage_margin_threshold": args.query_coverage_margin_threshold,
        "query_coverage_hard_query_fraction": float(args.query_coverage_hard_query_fraction),
        "query_coverage_reserved_count": int(payload.get("query_coverage_reserved_count", 0)),
        "query_coverage": dict(payload.get("query_coverage_metadata", {})),
        "strict_support_cache": str(args.strict_support_cache),
        "strict_support_fraction": float(args.strict_support_fraction),
        "strict_support_score_threshold": float(args.strict_support_score_threshold),
        "strict_support_reprojection_threshold_px": args.strict_support_reprojection_threshold_px,
        "strict_support_reserved_count": int(payload.get("strict_support_reserved_count", 0)),
        "strict_support": dict(payload.get("strict_support_metadata", {})),
        "protected_source_map": str(args.protected_source_map),
        "protected_source_idx_path": str(args.protected_source_idx_path),
        "protected_source_fraction": float(payload.get("protected_source_fraction", 0.0)),
        "protected_source_reserved_count": int(payload.get("protected_source_reserved_count", 0)),
        "protected_source": dict(payload.get("protected_source_metadata", {})),
        "locability_blend": float(args.locability_blend),
        "seed": int(args.seed),
        "residual_feedback_enabled": False,
        "selector_feedback_enabled": True,
        "rho_feedback_enabled": False,
        "single_path_deployment": True,
        "branch_selection": False,
    }


def _load_protected_source_idx(args: argparse.Namespace) -> tuple[torch.Tensor | None, dict[str, Any] | None, float]:
    protected_map = Path(args.protected_source_map) if str(args.protected_source_map) else None
    protected_idx_path = Path(args.protected_source_idx_path) if str(args.protected_source_idx_path) else None
    if protected_map is not None and protected_idx_path is not None:
        raise ValueError("use only one of --protected_source_map or --protected_source_idx_path")
    if protected_map is not None:
        protected_idx_path = protected_map / "detector" / "sampled_idx.pkl"
    if protected_idx_path is None:
        return None, None, 0.0
    if not protected_idx_path.exists():
        raise FileNotFoundError(f"missing protected sampled_idx.pkl: {protected_idx_path}")
    with protected_idx_path.open("rb") as handle:
        protected_idx = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1).cpu()
    metadata = {
        "idx_path": str(protected_idx_path),
        "map": str(protected_map) if protected_map is not None else "",
    }
    fraction = 1.0 if args.protected_source_fraction is None else float(args.protected_source_fraction)
    return protected_idx, metadata, fraction


def _load_source_sampled_idx(source_map: Path) -> torch.Tensor:
    sampled_idx_path = source_map / "detector" / "sampled_idx.pkl"
    if not sampled_idx_path.exists():
        raise FileNotFoundError(f"missing sampled_idx.pkl: {sampled_idx_path}")
    with sampled_idx_path.open("rb") as handle:
        return torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1).cpu()


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        command = " ".join(shlex.quote(part) for part in sys.argv)
        args = build_argparser().parse_args()
    else:
        command = _command_from_args(args)
    source_map = Path(args.source_map)
    checkpoint_path = Path(args.checkpoint_path)
    output_map = Path(args.output_map)
    source_ply = _latest_point_cloud_path(source_map) or (source_map / "input.ply")
    if not source_ply.exists():
        raise FileNotFoundError(f"source map has no point cloud or input.ply: {source_map}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    xyz = _load_xyz(source_ply)
    selector = selector_from_checkpoint(checkpoint, num_gaussians=int(xyz.shape[0]))
    hard_negative_risk = None
    hard_negative_metadata: dict[str, Any] | None = None
    if str(args.hard_negative_cache):
        hard_negative_risk, hard_negative_metadata = hard_negative_risk_from_episode_cache(
            args.hard_negative_cache,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.hard_negative_score_threshold),
            reprojection_threshold_px=args.hard_negative_reprojection_threshold_px,
        )
    positive_support = None
    positive_support_metadata: dict[str, Any] | None = None
    if str(args.positive_support_cache):
        positive_support, positive_support_metadata = positive_support_from_episode_cache(
            args.positive_support_cache,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.positive_support_score_threshold),
            reprojection_threshold_px=args.positive_support_reprojection_threshold_px,
        )
    pose_information = None
    pose_information_metadata: dict[str, Any] | None = None
    if str(args.pose_information_cache):
        pose_information, pose_information_metadata = pose_information_from_episode_cache(
            args.pose_information_cache,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.pose_information_score_threshold),
            reprojection_threshold_px=args.pose_information_reprojection_threshold_px,
        )
    hard_query_support = None
    hard_query_support_metadata: dict[str, Any] | None = None
    if str(args.hard_query_support_cache):
        hard_query_support, hard_query_support_metadata = hard_query_support_from_episode_cache(
            args.hard_query_support_cache,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.hard_query_support_score_threshold),
            reprojection_threshold_px=args.hard_query_support_reprojection_threshold_px,
        )
    source_sampled_idx = _load_source_sampled_idx(source_map)
    support_guard_idx = None
    support_guard_metadata: dict[str, Any] | None = None
    if str(args.support_guard_cache):
        support_guard_idx, support_guard_metadata = native_support_reservation_from_episode_cache(
            args.support_guard_cache,
            source_idx=source_sampled_idx,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.support_guard_score_threshold),
            reprojection_threshold_px=args.support_guard_reprojection_threshold_px,
            margin_threshold=args.support_guard_margin_threshold,
            hard_query_fraction=float(args.support_guard_hard_query_fraction),
            preserve_counts=bool(args.support_guard_preserve_counts),
        )
    query_coverage_idx = None
    query_coverage_metadata: dict[str, Any] | None = None
    if str(args.query_coverage_cache):
        query_coverage_idx, query_coverage_metadata = query_coverage_reservation_from_episode_cache(
            args.query_coverage_cache,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.query_coverage_score_threshold),
            reprojection_threshold_px=args.query_coverage_reprojection_threshold_px,
            margin_threshold=args.query_coverage_margin_threshold,
            hard_query_fraction=float(args.query_coverage_hard_query_fraction),
        )
    strict_support = None
    strict_support_metadata: dict[str, Any] | None = None
    if str(args.strict_support_cache):
        strict_support, strict_support_metadata = positive_support_from_episode_cache(
            args.strict_support_cache,
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.strict_support_score_threshold),
            reprojection_threshold_px=args.strict_support_reprojection_threshold_px,
        )
    protected_source_idx, protected_source_metadata, protected_source_fraction = _load_protected_source_idx(args)
    payload = resample_detector_landmarks(
        source_detector_dir=source_map / "detector",
        selector=selector,
        xyz=xyz,
        budget=_budget_value(args.budget),
        selector_weight=float(args.selector_weight),
        source_score_weight=float(args.source_score_weight),
        selector_transform=str(args.selector_transform),
        coverage_grid=int(args.coverage_grid),
        candidate_pool=args.candidate_pool,
        preserve_source_order=bool(args.preserve_source_order),
        source_retention_fraction=float(args.source_retention_fraction),
        hard_negative_risk=hard_negative_risk,
        hard_negative_weight=float(args.hard_negative_weight),
        hard_negative_metadata=hard_negative_metadata,
        positive_support=positive_support,
        positive_support_weight=float(args.positive_support_weight),
        positive_support_metadata=positive_support_metadata,
        pose_information=pose_information,
        pose_information_weight=float(args.pose_information_weight),
        pose_information_metadata=pose_information_metadata,
        hard_query_support=hard_query_support,
        hard_query_support_weight=float(args.hard_query_support_weight),
        hard_query_support_metadata=hard_query_support_metadata,
        support_guard_idx=support_guard_idx,
        support_guard_fraction=float(args.support_guard_fraction),
        support_guard_metadata=support_guard_metadata,
        query_coverage_idx=query_coverage_idx,
        query_coverage_fraction=float(args.query_coverage_fraction),
        query_coverage_metadata=query_coverage_metadata,
        strict_support=strict_support,
        strict_support_fraction=float(args.strict_support_fraction),
        strict_support_metadata=strict_support_metadata,
        protected_source_idx=protected_source_idx,
        protected_source_fraction=float(protected_source_fraction),
        protected_source_metadata=protected_source_metadata,
    )
    support_audit: dict[str, Any] | None = None
    if str(args.support_guard_cache):
        support_audit = support_preservation_audit_from_episode_cache(
            args.support_guard_cache,
            source_idx=source_sampled_idx,
            candidate_idx=payload["sampled_idx"],
            num_gaussians=int(xyz.shape[0]),
            base_gaussian_id=checkpoint.get("base_gaussian_id"),
            score_threshold=float(args.support_guard_score_threshold),
            reprojection_threshold_px=args.support_guard_reprojection_threshold_px,
            margin_threshold=args.support_guard_margin_threshold,
            hard_query_fraction=float(args.support_guard_hard_query_fraction),
        )
        payload["support_audit"] = support_audit
    manifest = _resampling_manifest(args, payload, source_ply, command=command)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    build_unified_lff_map(
        source_map=source_map,
        output_map=output_map,
        checkpoint_path=checkpoint_path,
        overwrite=not bool(args.no_overwrite),
        gate_locability_blend=float(args.locability_blend),
        descriptor_mode=args.descriptor_mode,
        gate_transform="identity",
        apply_to_detector_scores=False,
        apply_to_ply_locability=True,
        ablation_type="selector_resampled",
    )
    write_resampled_detector_payload(output_map / "detector", payload)
    (output_map / "selector_resampling_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_map / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if support_audit is not None:
        (output_map / "support_audit.json").write_text(
            json.dumps(support_audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(
        "[selector_resampling_export] wrote "
        f"{output_map} ({payload['source_count']} -> {payload['output_count']} sampled landmarks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
