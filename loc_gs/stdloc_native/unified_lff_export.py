from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.lff_export import (
    _load_ply_loc_features,
    _latest_point_cloud_path,
    _source_locability_logits,
    restore_descriptor_norms,
    write_lff_point_cloud,
)
from loc_gs.stdloc_native.soft_prior import _dump_pickle, _load_pickle, _mirror_map, _reset_path


def _checkpoint_descriptors(checkpoint: dict[str, Any]) -> torch.Tensor:
    if "export_descriptors" not in checkpoint:
        raise KeyError("unified LFF checkpoint is missing export_descriptors")
    desc = torch.as_tensor(checkpoint["export_descriptors"], dtype=torch.float32)
    if desc.dim() != 2:
        raise ValueError("export_descriptors must have shape [N,D]")
    return desc


def _checkpoint_gaussian_ids(checkpoint: dict[str, Any], *, count: int, full_count: int) -> torch.Tensor:
    ids = torch.as_tensor(checkpoint.get("base_gaussian_id", torch.empty(0)), dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        if int(count) != int(full_count):
            raise ValueError("checkpoint has no base_gaussian_id and does not cover the full source map")
        ids = torch.arange(full_count, dtype=torch.long)
    if ids.numel() != int(count):
        raise ValueError("base_gaussian_id must have one id per exported descriptor")
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(full_count)):
        raise IndexError("base_gaussian_id contains ids outside the source map")
    return ids


def _checkpoint_gate(checkpoint: dict[str, Any], *, count: int) -> torch.Tensor | None:
    if "gate" not in checkpoint:
        return None
    gate = torch.as_tensor(checkpoint["gate"], dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)
    if gate.shape[0] != int(count):
        raise ValueError("gate must have one value per exported descriptor")
    return gate


def _write_updated_ply(
    source_ply: Path,
    output_ply: Path,
    exported_desc: torch.Tensor,
    gaussian_ids: torch.Tensor,
    *,
    locability_logits: torch.Tensor | None = None,
    allow_missing_loc_fields: bool = False,
) -> bool:
    try:
        raw = _load_ply_loc_features(source_ply)
    except ValueError as exc:
        if allow_missing_loc_fields and "no loc_*" in str(exc):
            return False
        raise
    if exported_desc.shape[1] != raw.shape[1]:
        raise ValueError(
            f"export descriptor dim {exported_desc.shape[1]} does not match source PLY dim {raw.shape[1]}"
        )
    full = raw.clone()
    restored = restore_descriptor_norms(exported_desc, raw[gaussian_ids])
    full[gaussian_ids] = restored
    if output_ply.exists() or output_ply.is_symlink():
        _reset_path(output_ply)
    write_lff_point_cloud(
        source_ply=source_ply,
        output_ply=output_ply,
        descriptors=full,
        locability_logits=locability_logits,
    )
    return True


def _blend_gate_into_locability(
    source_ply: Path,
    *,
    gaussian_ids: torch.Tensor,
    gate: torch.Tensor | None,
    blend: float,
) -> torch.Tensor | None:
    if gate is None or float(blend) <= 0.0:
        return None
    raw = _load_ply_loc_features(source_ply)
    source_logits = _source_locability_logits(source_ply)
    if source_logits is None:
        prob = torch.full((raw.shape[0],), 0.5, dtype=torch.float32)
    else:
        prob = torch.sigmoid(source_logits.float().reshape(-1))
    weight = min(max(float(blend), 0.0), 1.0)
    prob[gaussian_ids] = ((1.0 - weight) * prob[gaussian_ids] + weight * gate).clamp(1e-4, 1.0 - 1e-4)
    return torch.logit(prob.clamp(1e-4, 1.0 - 1e-4))


def _blend_gate_into_detector_scores(
    source_map: Path,
    source_ply: Path,
    output_map: Path,
    *,
    gaussian_ids: torch.Tensor,
    gate: torch.Tensor | None,
    num_gaussians: int,
    blend: float,
) -> bool:
    if gate is None or float(blend) <= 0.0:
        return False
    sampled_idx_path = source_map / "detector" / "sampled_idx.pkl"
    scores_path = output_map / "detector" / "sampled_scores.pkl"
    if not sampled_idx_path.exists():
        return False
    sampled_idx = torch.as_tensor(_load_pickle(sampled_idx_path), dtype=torch.long).reshape(-1).cpu()
    source_logits = _source_locability_logits(source_ply)
    source_prob = (
        torch.sigmoid(source_logits.float().reshape(-1)).cpu()
        if source_logits is not None and int(source_logits.numel()) == int(num_gaussians)
        else None
    )
    if scores_path.exists():
        score_payload = _load_pickle(scores_path)
        if isinstance(score_payload, dict):
            scores = dict(score_payload)
            sampled_scores = torch.as_tensor(
                scores.get("sampled_scores", torch.zeros(sampled_idx.shape[0])),
                dtype=torch.float32,
            ).reshape(-1).cpu()
            score_avg = (
                torch.as_tensor(scores["score_avg"], dtype=torch.float32).reshape(-1).clone().cpu()
                if "score_avg" in scores
                else None
            )
        else:
            sampled_scores = torch.as_tensor(score_payload, dtype=torch.float32).reshape(-1).cpu()
            scores = {"sampled_scores": sampled_scores}
            score_avg = None
    else:
        if source_prob is not None:
            sampled_scores = source_prob[sampled_idx].clone()
            score_avg = source_prob.clone()
        else:
            sampled_scores = torch.ones(sampled_idx.shape[0], dtype=torch.float32)
            score_avg = None
        scores = {"sampled_scores": sampled_scores}
    if sampled_scores.shape[0] != sampled_idx.shape[0]:
        return False
    gaussian_ids = gaussian_ids.to(device="cpu", dtype=torch.long)
    gate = gate.to(device="cpu", dtype=torch.float32)
    full_gate = torch.full((int(num_gaussians),), float("nan"), dtype=torch.float32)
    full_gate[gaussian_ids] = gate
    sampled_gate = full_gate[sampled_idx]
    valid_sampled = torch.isfinite(sampled_gate)
    weight = min(max(float(blend), 0.0), 1.0)
    if bool(valid_sampled.any()):
        sampled_scores = sampled_scores.clone()
        sampled_scores[valid_sampled] = (
            (1.0 - weight) * sampled_scores[valid_sampled] + weight * sampled_gate[valid_sampled]
        ).clamp(0.0, 1.0)
        scores["sampled_scores"] = sampled_scores
    if score_avg is not None and score_avg.shape[0] == int(num_gaussians):
        score_avg[gaussian_ids] = ((1.0 - weight) * score_avg[gaussian_ids] + weight * gate).clamp(0.0, 1.0)
        scores["score_avg"] = score_avg
    if scores_path.is_symlink():
        scores_path.unlink()
    _dump_pickle(scores, scores_path)
    return bool(valid_sampled.any())


def build_unified_lff_map(
    *,
    source_map: str | Path,
    output_map: str | Path,
    checkpoint_path: str | Path,
    overwrite: bool = True,
    gate_locability_blend: float = 0.0,
    descriptor_mode: str = "checkpoint",
) -> dict[str, Any]:
    """Export a trained Unified LFF-v2 descriptor checkpoint to a STDLoc map."""

    source_map = Path(source_map)
    output_map = Path(output_map)
    checkpoint_path = Path(checkpoint_path)
    descriptor_mode = str(descriptor_mode)
    if descriptor_mode not in {"checkpoint", "native"}:
        raise ValueError(f"unsupported descriptor_mode: {descriptor_mode}")
    if not source_map.exists():
        raise FileNotFoundError(f"source map not found: {source_map}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("unified LFF checkpoint must be a dict")
    source_ply = _latest_point_cloud_path(source_map)
    if source_ply is None:
        source_ply = source_map / "input.ply"
    if not source_ply.exists():
        raise FileNotFoundError(f"source map has no point cloud or input.ply: {source_map}")
    raw = _load_ply_loc_features(source_ply)
    exported = _checkpoint_descriptors(checkpoint)
    gaussian_ids = _checkpoint_gaussian_ids(checkpoint, count=int(exported.shape[0]), full_count=int(raw.shape[0]))
    if descriptor_mode == "native":
        exported = torch.nn.functional.normalize(raw[gaussian_ids].float(), p=2, dim=-1)
    gate = _checkpoint_gate(checkpoint, count=int(exported.shape[0]))
    locability_logits = _blend_gate_into_locability(
        source_ply,
        gaussian_ids=gaussian_ids,
        gate=gate,
        blend=float(gate_locability_blend),
    )
    _mirror_map(source_map, output_map, overwrite=overwrite)
    output_ply = output_map / source_ply.relative_to(source_map)
    _write_updated_ply(source_ply, output_ply, exported, gaussian_ids, locability_logits=locability_logits)
    detector_scores_updated = _blend_gate_into_detector_scores(
        source_map,
        source_ply,
        output_map,
        gaussian_ids=gaussian_ids,
        gate=gate,
        num_gaussians=int(raw.shape[0]),
        blend=float(gate_locability_blend),
    )
    root_input_updated = False
    root_input = source_map / "input.ply"
    if root_input.exists() and root_input.resolve() != source_ply.resolve():
        root_input_updated = _write_updated_ply(
            root_input,
            output_map / "input.ply",
            exported,
            gaussian_ids,
            locability_logits=locability_logits,
            allow_missing_loc_fields=True,
        )
    manifest = {
        "source_map": str(source_map),
        "output_map": str(output_map),
        "checkpoint_path": str(checkpoint_path),
        "updated_gaussians": int(gaussian_ids.numel()),
        "source_gaussians": int(raw.shape[0]),
        "descriptor_dim": int(raw.shape[1]),
        "single_path_deployment": True,
        "branch_selection": False,
        "method": "unified_lff_v2_export_aligned",
        "descriptor_mode": descriptor_mode,
        "root_input_updated": bool(root_input_updated),
        "gate_locability_blend": float(gate_locability_blend),
        "detector_scores_updated": bool(detector_scores_updated),
    }
    (output_map / "unified_lff_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
