#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from plyfile import PlyData

from loc_gs.reporting.artifact_audit import artifact_split_audit, write_artifact_audit_bundle
from loc_gs.stdloc_native.soft_prior import _latest_point_cloud_path, _mirror_map, _write_point_cloud_locability


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


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv) if sys.argv else ""


def _load_dense_lsf_target(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload: Any = torch.load(Path(path), map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        raw = None
        for key in ("dense_lsf_target", "tensor", "values", "data"):
            if key in payload:
                raw = payload[key]
                break
        if raw is None:
            raise KeyError(f"{path} does not contain dense_lsf_target")
        if isinstance(payload.get("metadata", {}), dict):
            metadata = dict(payload["metadata"])
        target = torch.as_tensor(raw, dtype=torch.float32).reshape(-1).cpu()
    else:
        target = torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    if target.numel() == 0:
        raise ValueError("dense LSF target is empty")
    if not torch.isfinite(target).all():
        raise ValueError("dense LSF target contains non-finite values")
    return target.clamp(0.0, 1.0), metadata


def _split_is_test(split_name: str) -> bool:
    split = str(split_name).strip().lower()
    return split == "test" or split.startswith("test_") or split.endswith("_test")


def _vertex_count(path: Path) -> int:
    return int(PlyData.read(str(path))["vertex"].count)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a native STDLoc map with fixed dense LSF locability, without changing sparse sampled_idx."
    )
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--dense_lsf_target_path", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_map = Path(args.source_map)
    output_map = Path(args.output_map)
    dense_target_path = Path(args.dense_lsf_target_path)
    target, metadata = _load_dense_lsf_target(dense_target_path)
    split_name = str(metadata.get("split_name", metadata.get("split", "unknown")))
    if _split_is_test(split_name):
        raise ValueError("test split dense LSF targets are not allowed")

    source_point_cloud = _latest_point_cloud_path(source_map)
    if source_point_cloud is None:
        raise FileNotFoundError(f"source map has no point_cloud/iteration_*/point_cloud.ply: {source_map}")
    vertex_count = _vertex_count(source_point_cloud)
    if int(target.numel()) != vertex_count:
        raise ValueError(
            f"dense LSF target length {int(target.numel())} does not match source point cloud vertices {vertex_count}"
        )

    scene = str(args.scene or metadata.get("scene") or source_map.name)
    command = _command_from_argv()
    split_audit = artifact_split_audit(metadata, branch_selection=False)
    manifest = {
        "method": "loc_gs_dense_lsf_fixed_locability_prior",
        "git_commit": _git_commit(),
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "command": command,
        "scene": scene,
        "split_name": split_name,
        "data_root": "unknown",
        "checkpoint_path": "unknown",
        "map_path": str(source_map),
        "source_map": str(source_map),
        "output_map": str(output_map),
        "dense_lsf_target_path": str(dense_target_path),
        "single_path_deployment": True,
        "branch_selection": False,
        "same_budget": True,
        "sparse_selector_safe": False,
        "dense_support": {
            "enabled": True,
            "source": "dense_lsf_target",
            "usage_scope": str(metadata.get("usage_scope", "dense_residual_teacher_only")),
            "updated_point_cloud": str(source_point_cloud.relative_to(source_map)),
        },
        "source_artifact_metadata": metadata,
        "split_audit": split_audit,
    }
    metrics_summary = {
        "dense_locability_written": True,
        "gaussian_count": int(vertex_count),
        "target_mean": float(target.mean().item()),
        "target_min": float(target.min().item()),
        "target_max": float(target.max().item()),
        "same_budget": True,
    }

    _mirror_map(source_map, output_map, overwrite=bool(args.overwrite))
    output_point_cloud = output_map / source_point_cloud.relative_to(source_map)
    if output_point_cloud.exists() or output_point_cloud.is_symlink():
        output_point_cloud.unlink()
    _write_point_cloud_locability(source_point_cloud, output_point_cloud, target)
    (output_map / "dense_lsf_locability_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_artifact_audit_bundle(
        output_map,
        manifest=manifest,
        command=command,
        metrics_summary=metrics_summary,
        split_audit=split_audit,
    )
    print(json.dumps({"output_map": str(output_map), "gaussian_count": int(vertex_count)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
