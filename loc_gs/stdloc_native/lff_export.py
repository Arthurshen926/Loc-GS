from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from loc_gs.localization.descriptor_blend import gated_residual_descriptor_blend
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian, SuperPointOutputHead
from loc_gs.scripts.train_cambridge_hybrid import decode_gaussian_center_descriptors
from loc_gs.stdloc_native.soft_prior import (
    _latest_point_cloud_path,
    _load_pickle,
    _load_tensor_payload,
    _mirror_map,
    _reset_path,
    _write_soft_prior_cfg,
    build_soft_prior_map,
    load_selfmap_reliability,
    rank_normalize,
)


def protected_lff_descriptors(
    ply_descriptors: torch.Tensor,
    hybrid_descriptors: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    alpha_max: float = 0.03,
    reliability: float = 1.0,
) -> torch.Tensor:
    """Export a bounded LFF residual inside the STDLoc descriptor trust region."""

    alpha = max(float(alpha_max), 0.0) * min(max(float(reliability), 0.0), 1.0)
    return gated_residual_descriptor_blend(
        ply_descriptors,
        hybrid_descriptors,
        gate=gate,
        alpha_max=alpha,
    )


def restore_descriptor_norms(
    descriptors: torch.Tensor,
    source_descriptors: torch.Tensor,
) -> torch.Tensor:
    direction = F.normalize(torch.as_tensor(descriptors, dtype=torch.float32), p=2, dim=-1)
    source = torch.as_tensor(source_descriptors, dtype=torch.float32)
    norms = source.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return direction * norms


def _clamp01(values: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0)


def _full_matchability_from_sampled(
    *,
    locability: torch.Tensor,
    calibrated_matchability: torch.Tensor | None,
    sampled_idx: torch.Tensor | None,
) -> torch.Tensor | None:
    if calibrated_matchability is None or sampled_idx is None:
        return None
    sampled = _clamp01(calibrated_matchability)
    ids = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1)
    if sampled.shape[0] != ids.shape[0]:
        raise ValueError(
            f"calibrated matchability length {sampled.shape[0]} does not match sampled_idx {ids.shape[0]}"
        )
    full = locability.clone()
    full[ids] = rank_normalize(sampled)
    return full.clamp(0.0, 1.0)


def build_unified_selector(
    *,
    locability: torch.Tensor,
    calibrated_matchability: torch.Tensor | None = None,
    sampled_idx: torch.Tensor | None = None,
    source_scores: torch.Tensor | None = None,
    mode: str = "combined",
    matchability_weight: float = 0.5,
    source_weight: float = 0.0,
    floor: float = 0.0,
    power: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Build the per-Gaussian selector g_i for the unified LFF representation."""

    loc = _clamp01(locability)
    mode = str(mode)
    if mode not in {"uniform", "locability", "matchability", "combined", "reliability_boost"}:
        raise ValueError(f"unsupported selector mode: {mode}")
    match = _full_matchability_from_sampled(
        locability=loc,
        calibrated_matchability=calibrated_matchability,
        sampled_idx=sampled_idx,
    )
    if mode == "uniform":
        selector = torch.ones_like(loc)
    elif mode == "locability":
        selector = loc
    elif mode == "matchability":
        selector = match if match is not None else loc
    elif mode == "reliability_boost":
        if match is None:
            selector = loc
        else:
            w = min(max(float(matchability_weight), 0.0), 1.0)
            selector = (loc + w * (match - loc).clamp_min(0.0)).clamp(0.0, 1.0)
    else:
        if match is None:
            selector = loc
        else:
            w = min(max(float(matchability_weight), 0.0), 1.0)
            selector = ((1.0 - w) * loc + w * match).clamp(0.0, 1.0)

    if source_scores is not None and float(source_weight) > 0.0:
        src = rank_normalize(torch.as_tensor(source_scores, dtype=torch.float32).reshape(-1))
        if src.shape[0] != selector.shape[0]:
            raise ValueError(f"source score length {src.shape[0]} does not match selector {selector.shape[0]}")
        w = min(max(float(source_weight), 0.0), 1.0)
        selector = ((1.0 - w) * selector + w * src).clamp(0.0, 1.0)

    exponent = max(float(power), 1e-6)
    base = selector.clamp(0.0, 1.0).pow(exponent)
    floor_value = min(max(float(floor), 0.0), 1.0)
    selector = (floor_value + (1.0 - floor_value) * base).clamp(0.0, 1.0)
    stats: dict[str, float | str] = {
        "selector_mode": mode,
        "selector_mean": float(selector.mean().item()) if selector.numel() else 0.0,
        "selector_min": float(selector.min().item()) if selector.numel() else 0.0,
        "selector_max": float(selector.max().item()) if selector.numel() else 0.0,
        "selector_floor": float(floor_value),
        "selector_power": float(exponent),
        "matchability_weight": float(matchability_weight),
        "source_weight": float(source_weight),
        "matchability_available": 1.0 if match is not None else 0.0,
    }
    return selector, stats


def _sorted_loc_names(names: tuple[str, ...]) -> list[str]:
    loc_names = [name for name in names if name.startswith("loc_")]
    return sorted(loc_names, key=lambda name: int(name.split("_")[-1]))


def _load_ply_loc_features(path: Path) -> torch.Tensor:
    data = PlyData.read(str(path))["vertex"].data
    names = data.dtype.names or ()
    loc_names = _sorted_loc_names(names)
    if not loc_names:
        raise ValueError(f"{path} has no loc_* descriptor fields")
    values = np.stack([np.asarray(data[name]) for name in loc_names], axis=1)
    return torch.as_tensor(values, dtype=torch.float32)


def write_lff_point_cloud(
    *,
    source_ply: str | Path,
    output_ply: str | Path,
    descriptors: torch.Tensor,
    locability_logits: torch.Tensor | None = None,
) -> None:
    source_ply = Path(source_ply)
    output_ply = Path(output_ply)
    ply = PlyData.read(str(source_ply))
    vertex = ply["vertex"]
    data = vertex.data
    names = data.dtype.names or ()
    loc_names = _sorted_loc_names(names)
    desc = torch.as_tensor(descriptors, dtype=torch.float32).detach().cpu()
    if desc.dim() != 2 or desc.shape[0] != data.shape[0] or desc.shape[1] != len(loc_names):
        raise ValueError(
            f"descriptor shape {tuple(desc.shape)} must be "
            f"({data.shape[0]}, {len(loc_names)}) for {source_ply}"
        )
    new_data = data.copy()
    desc_np = desc.numpy().astype("f4", copy=False)
    for idx, name in enumerate(loc_names):
        new_data[name] = desc_np[:, idx]
    if locability_logits is not None:
        logits = torch.as_tensor(locability_logits, dtype=torch.float32).detach().cpu().reshape(-1)
        if logits.shape[0] != data.shape[0]:
            raise ValueError(
                f"locability shape {tuple(logits.shape)} must be ({data.shape[0]},) for {source_ply}"
            )
        if "locability_logit" not in names:
            new_dtype = data.dtype.descr + [("locability_logit", "f4")]
            expanded = np.empty(data.shape, dtype=new_dtype)
            for name in names:
                expanded[name] = new_data[name]
            new_data = expanded
        new_data["locability_logit"] = logits.numpy().astype("f4", copy=False)
    elements = [PlyElement.describe(new_data, vertex.name)]
    elements.extend(element for element in ply.elements if element.name != vertex.name)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order, comments=ply.comments).write(str(output_ply))


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    if not isinstance(args, dict):
        raise ValueError("LFF checkpoint args must be a dict-like object")
    return args


def _arg(args: dict[str, Any], name: str, default: Any) -> Any:
    value = args.get(name, default)
    return default if value is None else value


def load_lff_model_and_head(
    checkpoint_path: str | Path,
    *,
    ply_path: str | Path | None = None,
    device: str | torch.device = "cuda",
) -> tuple[HybridFeatureGaussian, SuperPointOutputHead, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"LFF checkpoint must be a dict: {checkpoint_path}")
    args = _checkpoint_args(checkpoint)
    source_ply = Path(ply_path or args.get("ply_path", ""))
    if not source_ply.exists():
        raise FileNotFoundError(f"LFF source PLY not found: {source_ply}")
    model = HybridFeatureGaussian(
        latent_dim=int(_arg(args, "latent_dim", 32)),
        hash_output_dim=int(_arg(args, "hash_output_dim", 48)),
        fine_dim=int(_arg(args, "fine_dim", 64)),
        coarse_dim=int(_arg(args, "coarse_dim", 64)),
        output_dim=int(_arg(args, "hybrid_output_dim", 128)),
    )
    model.load_from_ply(str(source_ply))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    sp_head = SuperPointOutputHead(
        fused_dim=int(_arg(args, "hybrid_output_dim", 128)),
        descriptor_dim=256,
        detector_dim=65,
        hidden_dim=256,
        num_res_blocks=2,
        use_3x3=True,
    ).to(device)
    sp_head.load_state_dict(checkpoint["sp_head_state_dict"])
    sp_head.eval()
    return model, sp_head, args


@torch.no_grad()
def decode_lff_center_descriptors(
    model: HybridFeatureGaussian,
    sp_head: SuperPointOutputHead,
    *,
    chunk_size: int = 16384,
) -> torch.Tensor:
    device = model.get_xyz().device
    n_points = int(model.num_gaussians)
    chunks = []
    for start in range(0, n_points, int(chunk_size)):
        end = min(start + int(chunk_size), n_points)
        ids = torch.arange(start, end, device=device, dtype=torch.long)
        chunks.append(decode_gaussian_center_descriptors(model, sp_head, ids).detach().cpu())
    return F.normalize(torch.cat(chunks, dim=0).float(), p=2, dim=-1)


def _source_locability_logits(source_ply: Path) -> torch.Tensor | None:
    data = PlyData.read(str(source_ply))["vertex"].data
    if "locability_logit" not in (data.dtype.names or ()):
        return None
    return torch.as_tensor(np.asarray(data["locability_logit"]), dtype=torch.float32)


def _load_sampled_idx(source_map: Path) -> torch.Tensor | None:
    path = source_map / "detector" / "sampled_idx.pkl"
    if not path.exists():
        return None
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1)


def _load_source_score_avg(source_map: Path, *, num_gaussians: int) -> torch.Tensor | None:
    path = source_map / "detector" / "sampled_scores.pkl"
    if not path.exists():
        return None
    payload = _load_pickle(path)
    if not isinstance(payload, dict) or "score_avg" not in payload:
        return None
    values = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1)
    if values.shape[0] != int(num_gaussians):
        return None
    return values


def _blend_locability_logits(
    source_logits: torch.Tensor | None,
    lff_logits: torch.Tensor,
    *,
    reliability: float,
    mode: str,
    selector: torch.Tensor | None = None,
    selector_weight: float = 0.0,
) -> torch.Tensor:
    rho = min(max(float(reliability), 0.0), 1.0)
    lff_prob = torch.sigmoid(lff_logits.float().reshape(-1))
    if source_logits is None:
        prob = lff_prob
    else:
        source_prob = torch.sigmoid(source_logits.float().reshape(-1))
        blended = ((1.0 - rho) * source_prob + rho * lff_prob).clamp(1e-4, 1.0 - 1e-4)
        if mode == "boost":
            prob = torch.maximum(source_prob, blended).clamp(1e-4, 1.0 - 1e-4)
        elif mode == "blend":
            prob = blended
        else:
            raise ValueError(f"unsupported locability fusion mode: {mode}")
    if selector is not None and float(selector_weight) > 0.0:
        sel = _clamp01(selector).clamp(1e-4, 1.0 - 1e-4)
        if sel.shape[0] != prob.shape[0]:
            raise ValueError(f"selector length {sel.shape[0]} does not match locability {prob.shape[0]}")
        w = min(max(float(selector_weight), 0.0), 1.0)
        if mode == "boost":
            prob = (prob + w * (sel - prob).clamp_min(0.0)).clamp(1e-4, 1.0 - 1e-4)
        elif mode == "blend":
            prob = ((1.0 - w) * prob + w * sel).clamp(1e-4, 1.0 - 1e-4)
        else:
            raise ValueError(f"unsupported locability fusion mode: {mode}")
    return torch.logit(prob.clamp(1e-4, 1.0 - 1e-4))


def build_lff_feature_map(
    *,
    source_map: str | Path,
    output_map: str | Path,
    checkpoint_path: str | Path,
    base_cfg_path: str | Path,
    output_cfg_path: str | Path,
    calibration_path: str | Path | None = None,
    rho: float | None = None,
    selfmap_reliability_path: str | Path | None = None,
    selfmap_reliability_stage: str = "dense",
    selfmap_reliability_center_cm: float = 10.0,
    selfmap_reliability_temperature_cm: float = 1.0,
    selfmap_reliability_r5_center: float | None = 0.5,
    selfmap_reliability_r5_temperature: float = 0.1,
    descriptor_alpha_max: float = 0.03,
    decode_chunk_size: int = 16384,
    selector_mode: str = "reliability_boost",
    selector_matchability_weight: float = 1.0,
    selector_source_weight: float = 0.0,
    selector_floor: float = 0.0,
    selector_power: float = 1.0,
    selector_locability_weight: float = 0.0,
    locability_fusion_mode: str = "boost",
    prior_blend: float = 0.25,
    score_fusion_mode: str = "boost",
    base_sparse_prior_weight: float = 0.0,
    base_dense_prior_weight: float = 0.05,
    overwrite: bool = True,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    source_map = Path(source_map)
    output_map = Path(output_map)
    base_cfg_path = Path(base_cfg_path)
    output_cfg_path = Path(output_cfg_path)
    checkpoint_path = Path(checkpoint_path)
    if rho is None:
        if selfmap_reliability_path is None:
            raise ValueError("rho or selfmap_reliability_path is required")
        reliability = load_selfmap_reliability(
            selfmap_reliability_path,
            stage=selfmap_reliability_stage,
            center_cm=selfmap_reliability_center_cm,
            temperature_cm=selfmap_reliability_temperature_cm,
            r5_center=selfmap_reliability_r5_center,
            r5_temperature=selfmap_reliability_r5_temperature,
        )
        rho = reliability["rho"]
    else:
        reliability = {"rho": float(rho)}
    rho = min(max(float(rho), 0.0), 1.0)

    if calibration_path is not None:
        manifest = build_soft_prior_map(
            source_map=source_map,
            output_map=output_map,
            calibration_path=calibration_path,
            base_cfg_path=base_cfg_path,
            output_cfg_path=output_cfg_path,
            rho=rho,
            prior_blend=prior_blend,
            fusion_mode=score_fusion_mode,
            base_sparse_prior_weight=base_sparse_prior_weight,
            base_dense_prior_weight=base_dense_prior_weight,
            update_point_cloud_locability=False,
            overwrite=overwrite,
        )
    else:
        _mirror_map(source_map, output_map, overwrite=overwrite)
        cfg = _write_soft_prior_cfg(
            base_cfg_path=base_cfg_path,
            output_cfg_path=output_cfg_path,
            rho=rho,
            base_sparse_prior_weight=base_sparse_prior_weight,
            base_dense_prior_weight=base_dense_prior_weight,
        )
        manifest = {
            "source_map": str(source_map),
            "output_map": str(output_map),
            "base_cfg_path": str(base_cfg_path),
            "output_cfg_path": str(output_cfg_path),
            "rho": float(rho),
            "sparse_prior_weight": float(cfg["sparse"]["landmark_prior_weight"]),
            "dense_prior_weight": float(cfg["dense"]["locability_prior_weight"]),
            "selfmap_reliability": reliability,
        }

    source_ply = _latest_point_cloud_path(source_map)
    if source_ply is None:
        raise FileNotFoundError(f"source map has no point cloud: {source_map}")
    output_ply = output_map / source_ply.relative_to(source_map)
    effective_alpha = max(float(descriptor_alpha_max), 0.0) * rho
    descriptor_written = False
    locability_written = False
    descriptor_stats: dict[str, float] = {}

    write_epsilon = 1e-6
    should_write_point_cloud = effective_alpha > write_epsilon or rho > write_epsilon
    if should_write_point_cloud:
        model, sp_head, _args = load_lff_model_and_head(
            checkpoint_path,
            ply_path=source_ply,
            device=device,
        )
        raw_ply_desc = _load_ply_loc_features(source_ply)
        ply_desc = F.normalize(raw_ply_desc, p=2, dim=-1)
        hybrid_desc = decode_lff_center_descriptors(
            model,
            sp_head,
            chunk_size=decode_chunk_size,
        )
        locability = model.get_locability().detach().float().squeeze(-1).cpu()
        sampled_idx = _load_sampled_idx(source_map)
        calibrated_matchability = (
            _load_tensor_payload(Path(calibration_path), "landmark_matchability")
            if calibration_path is not None
            else None
        )
        source_scores = _load_source_score_avg(source_map, num_gaussians=raw_ply_desc.shape[0])
        selector, selector_stats = build_unified_selector(
            locability=locability,
            calibrated_matchability=calibrated_matchability,
            sampled_idx=sampled_idx,
            source_scores=source_scores,
            mode=selector_mode,
            matchability_weight=selector_matchability_weight,
            source_weight=selector_source_weight,
            floor=selector_floor,
            power=selector_power,
        )
        exported_direction = protected_lff_descriptors(
            ply_desc.cpu(),
            hybrid_desc,
            gate=selector,
            alpha_max=descriptor_alpha_max,
            reliability=rho,
        )
        exported = restore_descriptor_norms(exported_direction, raw_ply_desc)
        lff_logits = model.get_locability_logits().detach().float().squeeze(-1).cpu()
        source_logits = _source_locability_logits(source_ply)
        locability_logits = _blend_locability_logits(
            source_logits,
            lff_logits,
            reliability=rho,
            mode=locability_fusion_mode,
            selector=selector,
            selector_weight=selector_locability_weight,
        )
        if output_ply.exists() or output_ply.is_symlink():
            _reset_path(output_ply)
        write_lff_point_cloud(
            source_ply=source_ply,
            output_ply=output_ply,
            descriptors=exported,
            locability_logits=locability_logits,
        )
        descriptor_written = effective_alpha > write_epsilon
        locability_written = rho > write_epsilon
        cos = F.cosine_similarity(ply_desc.cpu(), exported_direction, dim=-1)
        descriptor_stats = {
            "mean_cosine_to_ply": float(cos.mean().item()),
            "min_cosine_to_ply": float(cos.min().item()),
            "effective_alpha": float(effective_alpha),
            "mean_gate": float(selector.mean().item()),
            "mean_effective_gate": float((selector * rho).mean().item()),
            **selector_stats,
        }

    manifest.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "descriptor_alpha_max": float(descriptor_alpha_max),
            "effective_descriptor_alpha": float(effective_alpha),
            "locability_fusion_mode": locability_fusion_mode,
            "selector_mode": selector_mode,
            "selector_matchability_weight": float(selector_matchability_weight),
            "selector_source_weight": float(selector_source_weight),
            "selector_floor": float(selector_floor),
            "selector_power": float(selector_power),
            "selector_locability_weight": float(selector_locability_weight),
            "descriptor_written": bool(descriptor_written),
            "locability_written": bool(locability_written),
            "updated_point_cloud": str(output_ply.relative_to(output_map)) if locability_written else "",
            "lff_descriptor_stats": descriptor_stats,
            "selfmap_reliability": reliability,
        }
    )
    (output_map / "lff_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
