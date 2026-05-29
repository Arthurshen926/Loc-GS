#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from loc_gs.stdloc_native.lff_export import (
    _latest_point_cloud_path,
    _load_ply_loc_features,
    restore_descriptor_norms,
    write_lff_point_cloud,
)
from loc_gs.stdloc_native.soft_prior import _mirror_map, _reset_path


def _load_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("descriptor_artifact must contain a dict")
    return payload


def _source_descriptor_match_stats(
    *,
    artifact: dict[str, Any],
    source_descriptors: torch.Tensor,
    gaussian_ids: torch.Tensor,
    min_cosine: float,
    required: bool,
) -> dict[str, Any]:
    base = artifact.get("base_descriptors")
    if base is None:
        if required:
            raise ValueError(
                "descriptor artifact is not compatible with source map: "
                "missing base_descriptors for source descriptor match audit"
            )
        return {"required": False, "available": False}
    base_desc = F.normalize(torch.as_tensor(base, dtype=torch.float32).cpu(), p=2, dim=-1)
    if base_desc.shape != (gaussian_ids.numel(), source_descriptors.shape[1]):
        raise ValueError(
            "descriptor artifact is not compatible with source map: "
            "base_descriptors must match gaussian_ids and source descriptor dimension"
        )
    source = F.normalize(source_descriptors[gaussian_ids], p=2, dim=-1)
    cosine = (base_desc * source).sum(dim=-1).clamp(-1.0, 1.0)
    threshold = min(max(float(min_cosine), -1.0), 1.0)
    min_value = float(cosine.min().item()) if cosine.numel() else 1.0
    mean_value = float(cosine.mean().item()) if cosine.numel() else 1.0
    failing = int((cosine < threshold).sum().item()) if cosine.numel() else 0
    stats = {
        "required": bool(required),
        "available": True,
        "min_cosine_threshold": float(threshold),
        "min_cosine": min_value,
        "mean_cosine": mean_value,
        "failing_count": failing,
    }
    if failing > 0:
        raise ValueError(
            "descriptor artifact is not compatible with source map: "
            f"min_source_descriptor_cosine={min_value:.6f} < {threshold:.6f} "
            f"for {failing} / {int(cosine.numel())} rows"
        )
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize solver-weighted fused descriptors into a STDLoc map.")
    parser.add_argument("--source_map", required=True)
    parser.add_argument("--descriptor_artifact", required=True)
    parser.add_argument("--output_map", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_descriptor_min_cosine", type=float, default=0.995)
    parser.add_argument("--require_source_descriptor_match", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    source_map = Path(args.source_map)
    output_map = Path(args.output_map)
    artifact_path = Path(args.descriptor_artifact)
    artifact = _load_artifact(artifact_path)
    source_ply = _latest_point_cloud_path(source_map)
    if source_ply is None:
        raise FileNotFoundError(f"source map has no point_cloud/iteration_*/point_cloud.ply: {source_map}")
    raw_desc = _load_ply_loc_features(source_ply)
    fused_direction = F.normalize(torch.as_tensor(artifact["descriptors"], dtype=torch.float32).cpu(), p=2, dim=-1)
    gaussian_ids = torch.as_tensor(artifact["gaussian_ids"], dtype=torch.long).reshape(-1).cpu()
    if fused_direction.dim() != 2 or fused_direction.shape[0] != gaussian_ids.numel():
        raise ValueError("artifact descriptors must be [N,D] and match gaussian_ids")
    if fused_direction.shape[1] != raw_desc.shape[1]:
        raise ValueError("artifact descriptor dimension must match source PLY loc_* fields")
    if gaussian_ids.numel() and (int(gaussian_ids.min()) < 0 or int(gaussian_ids.max()) >= raw_desc.shape[0]):
        raise IndexError("artifact gaussian_ids are outside source PLY")
    match_stats = _source_descriptor_match_stats(
        artifact=artifact,
        source_descriptors=raw_desc,
        gaussian_ids=gaussian_ids,
        min_cosine=float(args.source_descriptor_min_cosine),
        required=bool(args.require_source_descriptor_match),
    )

    _mirror_map(source_map, output_map, overwrite=bool(args.overwrite))
    output_ply = output_map / source_ply.relative_to(source_map)
    full_desc = raw_desc.clone()
    restored = restore_descriptor_norms(fused_direction, raw_desc[gaussian_ids])
    full_desc[gaussian_ids] = restored
    if output_ply.exists() or output_ply.is_symlink():
        _reset_path(output_ply)
    write_lff_point_cloud(source_ply=source_ply, output_ply=output_ply, descriptors=full_desc)
    metadata = dict(artifact.get("metadata", {}))
    manifest = {
        "scene": str(args.scene),
        "source_map": str(source_map),
        "output_map": str(output_map),
        "source_ply": str(source_ply),
        "descriptor_artifact": str(artifact_path),
        "descriptor_mode": str(metadata.get("descriptor_mode", "solver_weighted_feature_fusion")),
        "updated_gaussian_count": int(gaussian_ids.numel()),
        "source_gaussian_count": int(raw_desc.shape[0]),
        "descriptor_dim": int(raw_desc.shape[1]),
        "artifact_metadata": metadata,
        "source_descriptor_match": match_stats,
        "same_budget": True,
        "single_path_deployment": True,
        "branch_selection": False,
    }
    manifest_path = output_map / "solver_weighted_feature_map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_map": str(output_map), "manifest": str(manifest_path), "updated_gaussian_count": int(gaussian_ids.numel())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
