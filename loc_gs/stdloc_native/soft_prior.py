from __future__ import annotations

import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml


def rank_normalize(values: torch.Tensor) -> torch.Tensor:
    flat = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if flat.numel() == 0:
        return flat.clone()
    finite = torch.isfinite(flat)
    if not finite.all():
        fill = flat[finite].min() if finite.any() else flat.new_tensor(0.0)
        flat = torch.where(finite, flat, fill)
    if flat.numel() == 1:
        return torch.ones_like(flat)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(flat.numel(), dtype=torch.float32)
    return ranks / float(flat.numel() - 1)


def selfmap_reliability_weight(
    median_te_cm: float,
    *,
    center_cm: float = 10.0,
    temperature_cm: float = 1.0,
) -> float:
    temperature = max(float(temperature_cm), 1e-6)
    x = (float(center_cm) - float(median_te_cm)) / temperature
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _increasing_sigmoid(value: float, *, center: float, temperature: float) -> float:
    temperature = max(float(temperature), 1e-6)
    x = (float(value) - float(center)) / temperature
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def _load_json(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"self-map reliability summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"self-map reliability summary must be a JSON object: {path}")
    return data


def _stage_value(stage: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = stage.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_selfmap_reliability(
    path_or_dir: str | Path,
    *,
    stage: str = "dense",
    center_cm: float = 10.0,
    temperature_cm: float = 1.0,
    r5_center: float | None = None,
    r5_temperature: float = 0.1,
) -> dict[str, float]:
    data = _load_json(path_or_dir)
    stage_data = data.get(stage, {})
    if not isinstance(stage_data, dict):
        raise ValueError(f"self-map reliability stage must be an object: {stage}")
    median = _stage_value(stage_data, "median_te_cm", "median_te")
    if median is None:
        raise ValueError(f"self-map reliability summary has no {stage}.median_te_cm")
    r5 = _stage_value(stage_data, "recall_5cm_5d", "recall_5cm_5deg")
    median_rho = selfmap_reliability_weight(
        median,
        center_cm=center_cm,
        temperature_cm=temperature_cm,
    )
    r5_rho = 1.0
    if r5_center is not None and r5 is not None:
        r5_rho = _increasing_sigmoid(
            float(r5),
            center=float(r5_center),
            temperature=r5_temperature,
        )
    return {
        "rho": float(median_rho * r5_rho),
        "median_rho": float(median_rho),
        "r5_rho": float(r5_rho),
        "median_te_cm": float(median),
        "recall_5cm_5d": float(r5) if r5 is not None else float("nan"),
    }


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _mirror_map(source_map: Path, output_map: Path, *, overwrite: bool) -> None:
    if output_map.exists() or output_map.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output map already exists: {output_map}")
        _reset_path(output_map)
    output_map.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_map.rglob("*")):
        rel = src.relative_to(source_map)
        dst = output_map / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())


def _load_tensor_payload(path: Path, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if key not in payload:
            raise KeyError(f"{path} has no {key}")
        payload = payload[key]
    return torch.as_tensor(payload, dtype=torch.float32).reshape(-1)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _latest_point_cloud_path(map_dir: Path) -> Path | None:
    point_cloud_root = map_dir / "point_cloud"
    if not point_cloud_root.exists():
        return None
    candidates = []
    for path in point_cloud_root.glob("iteration_*/point_cloud.ply"):
        try:
            iteration = int(path.parent.name.split("_")[-1])
        except ValueError:
            continue
        candidates.append((iteration, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _logit(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp(1e-4, 1.0 - 1e-4)
    return torch.log(values / (1.0 - values))


def _write_point_cloud_locability(source_ply: Path, output_ply: Path, locability: torch.Tensor) -> None:
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(str(source_ply))
    vertex = ply["vertex"]
    data = vertex.data
    logits = _logit(locability).cpu().numpy().astype("f4")
    if logits.shape[0] != data.shape[0]:
        raise ValueError(
            f"locability length {logits.shape[0]} does not match point cloud vertices {data.shape[0]}"
        )
    if "locability_logit" in data.dtype.names:
        new_data = data.copy()
        new_data["locability_logit"] = logits
    else:
        new_dtype = data.dtype.descr + [("locability_logit", "f4")]
        new_data = np.empty(data.shape, dtype=new_dtype)
        for name in data.dtype.names:
            new_data[name] = data[name]
        new_data["locability_logit"] = logits
    elements = [PlyElement.describe(new_data, vertex.name)]
    elements.extend(element for element in ply.elements if element.name != vertex.name)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order, comments=ply.comments).write(str(output_ply))


def _write_soft_prior_cfg(
    *,
    base_cfg_path: Path,
    output_cfg_path: Path,
    rho: float,
    base_sparse_prior_weight: float,
    base_dense_prior_weight: float,
) -> dict[str, Any]:
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"STDLoc cfg must be a YAML object: {base_cfg_path}")
    sparse = cfg.setdefault("sparse", {})
    dense = cfg.setdefault("dense", {})
    sparse["landmark_score_path"] = "detector/sampled_scores.pkl"
    sparse["landmark_prior_weight"] = float(base_sparse_prior_weight) * float(rho)
    dense["locability_prior_weight"] = float(base_dense_prior_weight) * float(rho)
    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    output_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def build_soft_prior_map(
    *,
    source_map: str | Path,
    output_map: str | Path,
    calibration_path: str | Path,
    base_cfg_path: str | Path,
    output_cfg_path: str | Path,
    rho: float | None = None,
    selfmap_reliability_path: str | Path | None = None,
    selfmap_reliability_stage: str = "dense",
    selfmap_reliability_center_cm: float = 10.0,
    selfmap_reliability_temperature_cm: float = 1.0,
    selfmap_reliability_r5_center: float | None = None,
    selfmap_reliability_r5_temperature: float = 0.1,
    prior_blend: float = 1.0,
    fusion_mode: str = "blend",
    base_sparse_prior_weight: float = 0.05,
    base_dense_prior_weight: float = 0.05,
    update_point_cloud_locability: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    source_map = Path(source_map)
    output_map = Path(output_map)
    calibration_path = Path(calibration_path)
    base_cfg_path = Path(base_cfg_path)
    output_cfg_path = Path(output_cfg_path)
    if not source_map.exists():
        raise FileNotFoundError(f"source STDLoc map not found: {source_map}")
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
    mix = min(max(float(prior_blend) * rho, 0.0), 1.0)
    fusion_mode = str(fusion_mode)
    if fusion_mode not in {"blend", "boost"}:
        raise ValueError(f"unsupported soft-prior fusion mode: {fusion_mode}")

    sampled_idx_path = source_map / "detector/sampled_idx.pkl"
    if not sampled_idx_path.exists():
        raise FileNotFoundError(f"STDLoc sampled_idx not found: {sampled_idx_path}")
    sampled_idx = torch.as_tensor(_load_pickle(sampled_idx_path), dtype=torch.long).reshape(-1)
    calibrated = rank_normalize(_load_tensor_payload(calibration_path, "landmark_matchability"))
    if calibrated.shape[0] != sampled_idx.shape[0]:
        raise ValueError(
            f"calibrated matchability length {calibrated.shape[0]} does not match sampled_idx {sampled_idx.shape[0]}"
        )

    source_scores_path = source_map / "detector/sampled_scores.pkl"
    source_scores: dict[str, Any] = {}
    source_sampled = torch.zeros_like(calibrated)
    source_full: torch.Tensor | None = None
    if source_scores_path.exists():
        loaded = _load_pickle(source_scores_path)
        if isinstance(loaded, dict):
            source_scores = dict(loaded)
            if "sampled_scores" in source_scores:
                source_sampled = torch.as_tensor(source_scores["sampled_scores"], dtype=torch.float32).reshape(-1)
            if "score_avg" in source_scores:
                source_full = torch.as_tensor(source_scores["score_avg"], dtype=torch.float32).reshape(-1).clone()
                if "sampled_scores" not in source_scores:
                    source_sampled = source_full[sampled_idx]
        else:
            source_sampled = torch.as_tensor(loaded, dtype=torch.float32).reshape(-1)
    if source_sampled.shape[0] != calibrated.shape[0]:
        raise ValueError(
            f"source sampled score length {source_sampled.shape[0]} does not match calibrated {calibrated.shape[0]}"
        )

    point_cloud_path = _latest_point_cloud_path(source_map)
    if source_full is None and point_cloud_path is not None and update_point_cloud_locability:
        from plyfile import PlyData

        n_vertices = len(PlyData.read(str(point_cloud_path))["vertex"].data)
        source_full = torch.zeros(n_vertices, dtype=torch.float32)

    blended_sampled = ((1.0 - mix) * source_sampled + mix * calibrated).clamp(0.0, 1.0)
    fused_sampled = (
        torch.maximum(source_sampled.clamp(0.0, 1.0), blended_sampled)
        if fusion_mode == "boost"
        else blended_sampled
    )
    fused_full = None
    if source_full is not None:
        fused_full = source_full.clone().float()
        calibrated_full = source_full.clone().float()
        calibrated_full[sampled_idx] = calibrated
        blended_full = ((1.0 - mix) * fused_full + mix * calibrated_full).clamp(0.0, 1.0)
        fused_full = (
            torch.maximum(fused_full.clamp(0.0, 1.0), blended_full)
            if fusion_mode == "boost"
            else blended_full
        )

    _mirror_map(source_map, output_map, overwrite=overwrite)
    output_scores = dict(source_scores)
    output_scores["sampled_scores"] = fused_sampled.cpu()
    if fused_full is not None:
        output_scores["score_avg"] = fused_full.cpu()
    _dump_pickle(output_scores, output_map / "detector/sampled_scores.pkl")
    cfg = _write_soft_prior_cfg(
        base_cfg_path=base_cfg_path,
        output_cfg_path=output_cfg_path,
        rho=rho,
        base_sparse_prior_weight=base_sparse_prior_weight,
        base_dense_prior_weight=base_dense_prior_weight,
    )

    point_cloud_rel = None
    if update_point_cloud_locability and point_cloud_path is not None and fused_full is not None:
        point_cloud_rel = point_cloud_path.relative_to(source_map)
        output_point_cloud = output_map / point_cloud_rel
        if output_point_cloud.exists() or output_point_cloud.is_symlink():
            _reset_path(output_point_cloud)
        _write_point_cloud_locability(point_cloud_path, output_point_cloud, fused_full)

    manifest = {
        "source_map": str(source_map),
        "output_map": str(output_map),
        "calibration_path": str(calibration_path),
        "base_cfg_path": str(base_cfg_path),
        "output_cfg_path": str(output_cfg_path),
        "rho": float(rho),
        "prior_blend": float(prior_blend),
        "fusion_mode": fusion_mode,
        "effective_mix": float(mix),
        "base_sparse_prior_weight": float(base_sparse_prior_weight),
        "base_dense_prior_weight": float(base_dense_prior_weight),
        "sparse_prior_weight": float(cfg["sparse"]["landmark_prior_weight"]),
        "dense_prior_weight": float(cfg["dense"]["locability_prior_weight"]),
        "updated_point_cloud": str(point_cloud_rel) if point_cloud_rel is not None else "",
        "selfmap_reliability": reliability,
        "score_stats": {
            "source_sampled_mean": float(source_sampled.mean().item()),
            "calibrated_rank_mean": float(calibrated.mean().item()),
            "fused_sampled_mean": float(fused_sampled.mean().item()),
        },
    }
    (output_map / "soft_prior_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
