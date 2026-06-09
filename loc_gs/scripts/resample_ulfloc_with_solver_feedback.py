#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from loc_gs.scripts.export_ulfloc_sparse_feedback import (
    _dataset_namespace,
    _git_commit,
    _git_status,
    _import_ulfloc,
    _load_masks,
    resolve_train_images_to_read,
    resolve_ulfloc_source_path_for_loader,
)
from loc_gs.stdloc_native.solver_admissibility import make_replacement_admissibility_checker
from loc_gs.stdloc_native.solver_coverage_coreset import (
    build_solver_coverage_tables,
    solver_coverage_local_edit,
)
from loc_gs.stdloc_native.negative_support_memory import normalize_negative_support_graph
from loc_gs.stdloc_native.sparse_solver_set_selection import (
    SparseSetSelectionConfig,
    SparseSetSignals,
    per_query_support_from_solver_constraints,
    select_sparse_solver_set_from_full_gaussians,
    sparse_validation_signals_from_pnp_profile,
)


def read_ply_vertex_count(path: Path) -> int:
    with Path(path).open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"could not find element vertex header in {path}")


def _iteration_dir(model_path: Path, iteration: int) -> Path:
    if iteration == -1:
        point_cloud_root = model_path / "point_cloud"
        iterations = []
        for child in point_cloud_root.glob("iteration_*"):
            try:
                iterations.append(int(child.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        if not iterations:
            raise FileNotFoundError(f"no point_cloud/iteration_* directory under {model_path}")
        iteration = max(iterations)
    return model_path / "point_cloud" / f"iteration_{iteration}"


def _load_solver_feedback(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"solver feedback must be a dict: {path}")
    if "landmark_weights" not in payload:
        raise KeyError("solver feedback artifact must contain landmark_weights")
    split_name = str(payload.get("split_name", payload.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use ULF-Loc solver feedback exported from test split")
    return payload


def validate_solver_feedback_matches_source_map(
    *,
    feedback_path: Path,
    source_model_path: Path,
    iteration: int,
) -> dict[str, Any]:
    payload = _load_solver_feedback(feedback_path)
    weights = torch.as_tensor(payload["landmark_weights"]).reshape(-1)
    ply_path = _iteration_dir(source_model_path, iteration) / "point_cloud.ply"
    source_count = read_ply_vertex_count(ply_path)
    if int(weights.numel()) != int(source_count):
        raise ValueError(
            f"solver feedback landmark_weights length {weights.numel()} "
            f"does not match source map vertex count {source_count}"
        )
    return {
        "source_gaussian_count": int(source_count),
        "feedback_weight_count": int(weights.numel()),
        "feedback_split_name": str(payload.get("split_name", payload.get("split", "unknown"))),
        "feedback_schema_version": str(payload.get("schema_version", "unknown")),
    }


def _replace_path_with_symlink(source: Path, target: Path, *, force: bool) -> None:
    if target.is_symlink() or target.exists():
        if not force:
            raise FileExistsError(f"target already exists; pass --force to replace: {target}")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_ulfloc_solver_feedback_output(
    *,
    source_model_path: Path,
    output_model_path: Path,
    feedback_path: Path | None,
    log_name: str,
    iteration: int,
    force: bool = False,
) -> dict[str, Path]:
    if feedback_path is not None:
        validation = validate_solver_feedback_matches_source_map(
            feedback_path=feedback_path,
            source_model_path=source_model_path,
            iteration=iteration,
        )
        del validation
    source_point_cloud = source_model_path / "point_cloud"
    if not source_point_cloud.exists():
        raise FileNotFoundError(f"source point_cloud directory not found: {source_point_cloud}")
    source_cfg_args = source_model_path / "cfg_args"
    if not source_cfg_args.exists():
        raise FileNotFoundError(f"source cfg_args not found: {source_cfg_args}")
    output_model_path.mkdir(parents=True, exist_ok=True)
    _replace_path_with_symlink(source_point_cloud, output_model_path / "point_cloud", force=force)
    shutil.copy2(source_cfg_args, output_model_path / "cfg_args")

    output_eval_dir = output_model_path / str(log_name)
    output_eval_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "point_cloud": output_model_path / "point_cloud",
        "output_eval_dir": output_eval_dir,
    }
    if feedback_path is not None:
        output_feedback = output_eval_dir / "solver_feedback.pkl"
        shutil.copy2(feedback_path, output_feedback)
        paths["solver_feedback"] = output_feedback
    return paths


def _import_ulfloc_postprocess(ulf_root: Path) -> dict[str, Any]:
    if str(ulf_root) not in sys.path:
        sys.path.insert(0, str(ulf_root))
    from encoders.feature_extractor import FeatureExtractor  # type: ignore
    from gaussian_renderer import get_render_visible_mask  # type: ignore
    from utils.gsfeature_fusion import feature_fusion  # type: ignore
    from utils.keypoints_sample import keypoints_vote  # type: ignore

    imports = _import_ulfloc(ulf_root)
    imports.update(
        {
            "FeatureExtractor": FeatureExtractor,
            "feature_fusion": feature_fusion,
            "get_render_visible_mask": get_render_visible_mask,
            "keypoints_vote": keypoints_vote,
        }
    )
    return imports


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_path_string(path: str | os.PathLike[str] | None) -> str | None:
    return None if path is None else str(Path(path))


def write_ulfloc_used_config(*, output_eval_dir: Path, cfg_path: Path, config: dict[str, Any]) -> Path:
    """Persist the exact ULF config used by this exported map/eval directory."""

    output_eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_eval_dir / Path(cfg_path).name
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def apply_solver_feedback_alpha_overrides(
    config: dict[str, Any],
    *,
    sampling_alpha: float | None = None,
    fusion_alpha: float | None = None,
) -> dict[str, Any]:
    solver_cfg = config.setdefault("solver_feedback", {})
    if not isinstance(solver_cfg, dict):
        raise TypeError("solver_feedback config must be a mapping")
    if sampling_alpha is not None:
        solver_cfg["sampling_alpha"] = float(sampling_alpha)
    if fusion_alpha is not None:
        solver_cfg["fusion_alpha"] = float(fusion_alpha)
    return config


def resolve_solver_feedback_path_for_resample(
    *,
    explicit_feedback_path: Path | str | None,
    source_model_path: Path,
    log_name: str,
    config: Mapping[str, Any],
) -> tuple[Path | None, str]:
    if explicit_feedback_path is not None:
        return Path(explicit_feedback_path), "explicit"
    feedback_config = config.get("solver_feedback", {}) if isinstance(config, Mapping) else {}
    if not isinstance(feedback_config, Mapping) or not bool(feedback_config.get("enabled", False)):
        return None, "disabled"
    raw_artifact_path = str(feedback_config.get("artifact_path", "")).strip()
    if not raw_artifact_path:
        return None, "disabled"
    artifact_path = Path(raw_artifact_path)
    inherited_path = artifact_path if artifact_path.is_absolute() else Path(source_model_path) / str(log_name) / artifact_path
    if not inherited_path.exists():
        raise FileNotFoundError(
            "solver_feedback is enabled in the ULF config but no artifact was provided or found at "
            f"the source eval directory: {inherited_path}"
        )
    return inherited_path, "inherited_source_eval"


def _load_solver_constraints(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("solver admissibility payload must be a JSON object")
    metadata = payload.get("metadata", {})
    split_name = str(payload.get("split_name", metadata.get("split_name", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use solver admissibility constraints from test split")
    return payload


def _load_descriptor_conflict_edges(path: Path | None) -> dict[int, dict[int, float]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("descriptor conflict graph must be a JSON object")
    metadata = payload.get("metadata", {})
    split_name = str(payload.get("split_name", metadata.get("split_name", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use descriptor conflict graph from test split")
    if "adjacency" in payload:
        raw_adjacency = payload.get("adjacency", {})
    else:
        raw_adjacency = normalize_negative_support_graph(payload).get("adjacency", {})
    if not isinstance(raw_adjacency, Mapping):
        return {}
    out: dict[int, dict[int, float]] = {}
    for raw_src, raw_neighbors in raw_adjacency.items():
        try:
            src = int(raw_src)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_neighbors, Mapping):
            continue
        for raw_dst, raw_value in raw_neighbors.items():
            try:
                dst = int(raw_dst)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if src < 0 or dst < 0 or dst == src or value <= 0.0:
                continue
            out.setdefault(src, {})[dst] = max(float(out.setdefault(src, {}).get(dst, 0.0)), float(value))
    return out


def _merge_conflict_edge_tables(*tables: Mapping[int, Mapping[int, float]]) -> dict[int, dict[int, float]]:
    merged: dict[int, dict[int, float]] = {}
    for table in tables:
        if not isinstance(table, Mapping):
            continue
        for raw_src, raw_neighbors in table.items():
            try:
                src = int(raw_src)
            except (TypeError, ValueError):
                continue
            if src < 0 or not isinstance(raw_neighbors, Mapping):
                continue
            for raw_dst, raw_value in raw_neighbors.items():
                try:
                    dst = int(raw_dst)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if dst < 0 or dst == src or value <= 0.0:
                    continue
                merged.setdefault(src, {})
                merged[src][dst] = max(float(merged[src].get(dst, 0.0)), float(value))
    return merged


def _load_sparse_validation_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"sparse validation profile not found: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise TypeError("sparse validation profile must be a JSON object")
    split_name = str(profile.get("split_name", profile.get("split", "unknown"))).strip()
    if split_name.lower() == "test":
        raise ValueError("test split sparse validation profile is not allowed")
    return profile


def descriptor_conflict_edges_from_sparse_validation_profile(
    profile: Mapping[str, Any],
    *,
    max_landmarks_per_query: int = 64,
) -> dict[int, dict[int, float]]:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    raw = profile.get("regression_per_query_risk_support", {})
    if not isinstance(raw, Mapping):
        return {}
    raw_protected = profile.get("protected_per_query_support", {})
    protected_by_query = raw_protected if isinstance(raw_protected, Mapping) else {}
    cap = max(0, int(max_landmarks_per_query))
    edges: dict[int, dict[int, float]] = {}

    def normalized_items(raw_support: Any) -> list[tuple[int, float]]:
        if not isinstance(raw_support, Mapping):
            return []
        items: list[tuple[int, float]] = []
        for raw_gid, raw_value in raw_support.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if gid < 0 or value <= 0.0:
                continue
            items.append((gid, value))
        items.sort(key=lambda item: (-float(item[1]), int(item[0])))
        if cap > 0:
            items = items[:cap]
        if not items:
            return []
        max_value = max(float(value) for _gid, value in items)
        if max_value <= 0.0:
            return []
        return [(int(gid), max(0.0, float(value) / max_value)) for gid, value in items]

    def add_edge(src: int, dst: int, value: float) -> None:
        if int(src) == int(dst) or float(value) <= 0.0:
            return
        edges.setdefault(int(src), {})
        edges[int(src)][int(dst)] = max(float(edges[int(src)].get(int(dst), 0.0)), float(value))
        edges.setdefault(int(dst), {})
        edges[int(dst)][int(src)] = max(float(edges[int(dst)].get(int(src), 0.0)), float(value))

    for raw_query_id, raw_support in raw.items():
        normalized = normalized_items(raw_support)
        if len(normalized) < 2:
            protected = normalized_items(protected_by_query.get(raw_query_id, {}))
            for src, src_value in normalized:
                for dst, dst_value in protected:
                    add_edge(src, dst, min(float(src_value), float(dst_value)))
            continue
        for first_index, (src, src_value) in enumerate(normalized[:-1]):
            for dst, dst_value in normalized[first_index + 1 :]:
                add_edge(src, dst, min(float(src_value), float(dst_value)))
        protected = normalized_items(protected_by_query.get(raw_query_id, {}))
        for src, src_value in normalized:
            for dst, dst_value in protected:
                add_edge(src, dst, min(float(src_value), float(dst_value)))
    return edges


def per_query_support_from_sparse_validation_profile(
    profile: Mapping[str, Any],
    *,
    support_scope: str = "all",
) -> dict[str, dict[int, float]]:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    scope = str(support_scope).strip().lower()
    if scope not in {"all", "protected_regressions"}:
        raise ValueError(f"unsupported sparse validation support scope: {support_scope!r}")
    protected_regression_queries = {
        str(query_id)
        for query_id in profile.get("protected_regression_query_ids", [])
        if str(query_id).strip()
    }
    out: dict[str, dict[int, float]] = {}
    for field_name in ("protected_per_query_support", "validated_per_query_support"):
        raw = profile.get(field_name, {})
        if not isinstance(raw, Mapping):
            continue
        for raw_query_id, raw_support in raw.items():
            if not isinstance(raw_support, Mapping):
                continue
            query_id = str(raw_query_id)
            if (
                scope == "protected_regressions"
                and field_name == "protected_per_query_support"
                and query_id not in protected_regression_queries
            ):
                continue
            for raw_gid, raw_value in raw_support.items():
                try:
                    gid = int(raw_gid)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or value <= 0.0:
                    continue
                out.setdefault(query_id, {})
                out[query_id][gid] = float(out[query_id].get(gid, 0.0) + value)
    return out


def per_query_observations_from_sparse_validation_profile(
    profile: Mapping[str, Any],
    *,
    support_scope: str = "all",
) -> dict[str, dict[int, dict[str, Any]]]:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    scope = str(support_scope).strip().lower()
    if scope not in {"all", "protected_regressions"}:
        raise ValueError(f"unsupported sparse validation support scope: {support_scope!r}")
    protected_regression_queries = {
        str(query_id)
        for query_id in profile.get("protected_regression_query_ids", [])
        if str(query_id).strip()
    }
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for field_name in ("protected_per_query_observations", "validated_per_query_observations"):
        raw = profile.get(field_name, {})
        if not isinstance(raw, Mapping):
            continue
        for raw_query_id, raw_observations in raw.items():
            if not isinstance(raw_observations, Mapping):
                continue
            query_id = str(raw_query_id)
            if (
                scope == "protected_regressions"
                and field_name == "protected_per_query_observations"
                and query_id not in protected_regression_queries
            ):
                continue
            for raw_gid, raw_observation in raw_observations.items():
                try:
                    gid = int(raw_gid)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or not isinstance(raw_observation, Mapping):
                    continue
                out.setdefault(query_id, {})[gid] = dict(raw_observation)
    return out


def sparse_validation_hard_reject_exempt_tensor_from_profile(
    profile: Mapping[str, Any],
    *,
    num_gaussians: int,
) -> torch.Tensor:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use sparse validation profile from test split")
    mask = torch.zeros(int(num_gaussians), dtype=torch.bool)
    protected_regression_queries = {
        str(query_id)
        for query_id in profile.get("protected_regression_query_ids", [])
        if str(query_id).strip()
    }
    raw = profile.get("protected_per_query_support", {})
    if not isinstance(raw, Mapping):
        return mask
    for raw_query_id, raw_support in raw.items():
        query_id = str(raw_query_id)
        if query_id not in protected_regression_queries or not isinstance(raw_support, Mapping):
            continue
        for raw_gid, raw_value in raw_support.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= gid < int(num_gaussians) and value > 0.0:
                mask[gid] = True
    return mask


def per_query_support_from_ray_feedback(payload: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    metadata = payload.get("metadata", {})
    split_name = str(payload.get("split_name", metadata.get("split_name", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use ray solver feedback from test split")
    raw = payload.get("per_query_support", {})
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, dict[int, float]] = {}
    for raw_query_id, raw_support in raw.items():
        if not isinstance(raw_support, Mapping):
            continue
        query_id = str(raw_query_id)
        for raw_gid, raw_value in raw_support.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if gid < 0 or value <= 0.0:
                continue
            out.setdefault(query_id, {})
            out[query_id][gid] = float(out[query_id].get(gid, 0.0) + value)
    return out


def per_query_observations_from_ray_feedback(payload: Mapping[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    metadata = payload.get("metadata", {})
    split_name = str(payload.get("split_name", metadata.get("split_name", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use ray solver feedback from test split")
    raw = payload.get("per_query_observations", {})
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for raw_query_id, raw_observations in raw.items():
        if not isinstance(raw_observations, Mapping):
            continue
        query_id = str(raw_query_id)
        for raw_gid, raw_observation in raw_observations.items():
            try:
                gid = int(raw_gid)
            except (TypeError, ValueError):
                continue
            if gid < 0 or not isinstance(raw_observation, Mapping):
                continue
            out.setdefault(query_id, {})[gid] = dict(raw_observation)
    return out


def merge_per_query_support(
    base: Mapping[str, Mapping[int, float]],
    extra: Mapping[str, Mapping[int, float]],
) -> dict[str, dict[int, float]]:
    merged: dict[str, dict[int, float]] = {}
    for table in (base, extra):
        for raw_query_id, raw_support in table.items():
            if not isinstance(raw_support, Mapping):
                continue
            query_id = str(raw_query_id)
            for raw_gid, raw_value in raw_support.items():
                try:
                    gid = int(raw_gid)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or value <= 0.0:
                    continue
                merged.setdefault(query_id, {})
                merged[query_id][gid] = float(merged[query_id].get(gid, 0.0) + value)
    return merged


def merge_per_query_observations(
    base: Mapping[str, Mapping[int, Mapping[str, Any]]],
    extra: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    merged: dict[str, dict[int, dict[str, Any]]] = {}
    for table in (base, extra):
        for raw_query_id, raw_observations in table.items():
            if not isinstance(raw_observations, Mapping):
                continue
            query_id = str(raw_query_id)
            for raw_gid, raw_observation in raw_observations.items():
                try:
                    gid = int(raw_gid)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or not isinstance(raw_observation, Mapping):
                    continue
                current = merged.setdefault(query_id, {}).setdefault(gid, {})
                current.update(dict(raw_observation))
    return merged


def full_set_needs_descriptor_features(args: argparse.Namespace) -> bool:
    return bool(
        int(getattr(args, "full_set_auto_descriptor_conflict_top_k", 0)) > 0
        or int(getattr(args, "full_set_sparse_validation_risk_expand_top_k", 0)) > 0
    )


def _load_ray_solver_feedback(path: Path, *, num_gaussians: int) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"ray solver feedback must be a dict: {path}")
    metadata = payload.get("metadata", {})
    split_name = str(payload.get("split_name", metadata.get("split_name", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("refusing to use ray solver feedback from test split")
    for key in ("support_score", "hard_negative_risk", "artifact_risk"):
        if key not in payload:
            continue
        tensor = torch.as_tensor(payload[key]).reshape(-1).cpu()
        if int(tensor.numel()) != int(num_gaussians):
            raise ValueError(
                f"{key} length {tensor.numel()} does not match source gaussian count {num_gaussians}"
            )
    return payload


def write_map_resample_audit_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    split_audit: dict[str, Any],
) -> None:
    """Write map-export audit files and a map-prefixed copy that eval will not overwrite."""

    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics_summary.json", metrics)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "map_manifest.json", manifest)
    _write_json(output_dir / "map_metrics_summary.json", metrics)
    _write_json(output_dir / "map_split_audit.json", split_audit)


def _sampled_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    try:
        return int(len(payload))
    except TypeError:
        return int(payload.numel())


def _load_sampled_idx(path: Path) -> torch.Tensor:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return torch.as_tensor(payload, dtype=torch.long).reshape(-1).cpu()


def prepare_reused_sampled_idx(
    *,
    source_sampled_idx: Path,
    output_eval_dir: Path,
    landmark_file_name: str,
    force: bool = False,
) -> dict[str, Any]:
    """Preload a sampled_idx artifact so ULF-Loc reuses the fixed subset."""

    source = Path(source_sampled_idx)
    if not source.is_file():
        raise FileNotFoundError(f"reuse sampled_idx source not found: {source}")
    target = Path(output_eval_dir) / str(landmark_file_name)
    if target.exists():
        if not force:
            raise FileExistsError(f"reused sampled_idx target already exists: {target}")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "reuse_sampled_idx": str(source),
        "reused_sampled_idx_target": str(target),
        "reused_sampled_count": int(_load_sampled_idx(target).numel()),
    }


def _preserve_source_descriptors_for_shared_landmarks(
    sampled_idx: torch.Tensor,
    features: torch.Tensor,
    source_sampled_idx: torch.Tensor,
    source_features: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    sampled = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    current = torch.as_tensor(features, dtype=torch.float32).clone()
    source_idx = torch.as_tensor(source_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    source = torch.as_tensor(source_features, dtype=torch.float32).to(device=current.device)
    if int(current.shape[0]) != int(sampled.numel()):
        raise ValueError("feature row count must match sampled_idx length")
    if int(source.shape[0]) != int(source_idx.numel()):
        raise ValueError("source feature row count must match source sampled_idx length")
    if current.ndim != source.ndim or tuple(current.shape[1:]) != tuple(source.shape[1:]):
        raise ValueError("source and target descriptor feature shapes are incompatible")
    source_pos = {int(gid): int(pos) for pos, gid in enumerate(source_idx.tolist())}
    replaced = 0
    for row, raw_gid in enumerate(sampled.tolist()):
        source_row = source_pos.get(int(raw_gid))
        if source_row is None:
            continue
        current[int(row)] = source[int(source_row)]
        replaced += 1
    return current, int(replaced)


def _load_optional_full_gaussian_vector(path: Path | None, *, num_gaussians: int, name: str) -> torch.Tensor:
    if path is None:
        return torch.zeros((int(num_gaussians),), dtype=torch.float32)
    parent_audit = Path(path).parent / "split_audit.json"
    if parent_audit.exists():
        audit = json.loads(parent_audit.read_text(encoding="utf-8"))
        split_name = str(audit.get("split_name", audit.get("checks", {}).get("query_split", {}).get("split_name", "")))
        if split_name.strip().lower() == "test":
            raise ValueError(f"refusing to use {name} from test split artifact: {path}")
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    tensor = torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    count = int(num_gaussians)
    if int(tensor.numel()) == count:
        return tensor
    if int(tensor.numel()) < count:
        out = torch.zeros((count,), dtype=torch.float32)
        out[: int(tensor.numel())] = tensor
        return out
    tail = tensor[count:]
    if bool((tail.abs() > 0.0).any()):
        raise ValueError(f"{name} length {tensor.numel()} exceeds source gaussian count {count} with nonzero tail")
    return tensor[:count]


def _load_query_match_dominance(
    path: Path | None,
    *,
    num_gaussians: int,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    if path is None:
        return {}, {}
    parent_audit = Path(path).parent / "split_audit.json"
    if parent_audit.exists():
        audit = json.loads(parent_audit.read_text(encoding="utf-8"))
        split_name = str(audit.get("split_name", audit.get("checks", {}).get("query_split", {}).get("split_name", "")))
        if split_name.strip().lower() == "test":
            raise ValueError(f"refusing to use query match dominance from test split artifact: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("query match dominance artifact must be a JSON object")
    split_name = str(payload.get("split_name", "")).strip().lower()
    if split_name == "test":
        raise ValueError("refusing to use query match dominance from test split")
    raw_queries = payload.get("queries", {})
    if not isinstance(raw_queries, Mapping):
        raise TypeError("query match dominance artifact must contain a queries object")

    def parse_table(field_name: str) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for raw_query_id, raw_entry in raw_queries.items():
            if not isinstance(raw_entry, Mapping):
                continue
            raw_scores = raw_entry.get(field_name, {})
            if not isinstance(raw_scores, Mapping):
                continue
            query_id = str(raw_query_id)
            for raw_gid, raw_value in raw_scores.items():
                try:
                    gid = int(raw_gid)
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if gid < 0 or gid >= int(num_gaussians) or value <= 0.0:
                    continue
                out.setdefault(query_id, {})
                out[query_id][gid] = float(out[query_id].get(gid, 0.0) + value)
        return out

    return parse_table("support_match_strength"), parse_table("match_competition_risk")


def _query_match_risk_reference_from_sampled_idx(
    per_query_match_competition_risk: Mapping[str, Mapping[int, float]],
    sampled_idx: torch.Tensor | None,
) -> dict[str, float]:
    if sampled_idx is None:
        return {}
    sampled_set = {int(gid) for gid in torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).tolist()}
    if not sampled_set:
        return {}
    out: dict[str, float] = {}
    for raw_query_id, risk_map in per_query_match_competition_risk.items():
        query_id = str(raw_query_id)
        total = 0.0
        for raw_gid, raw_value in risk_map.items():
            gid = int(raw_gid)
            if gid not in sampled_set:
                continue
            total += max(0.0, float(raw_value))
        out[query_id] = float(total)
    return out


def _query_match_competition_reference_landmarks_from_sampled_idx(
    per_query_match_competition_risk: Mapping[str, Mapping[int, float]],
    sampled_idx: torch.Tensor | None,
) -> dict[str, set[int]]:
    if sampled_idx is None:
        return {}
    sampled_set = {int(gid) for gid in torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).tolist()}
    if not sampled_set:
        return {}
    out: dict[str, set[int]] = {}
    for raw_query_id, risk_map in per_query_match_competition_risk.items():
        query_id = str(raw_query_id)
        for raw_gid, raw_value in risk_map.items():
            gid = int(raw_gid)
            if gid not in sampled_set or max(0.0, float(raw_value)) <= 0.0:
                continue
            out.setdefault(query_id, set()).add(gid)
    return out


def _optional_feedback_vector(
    payload: dict[str, Any],
    names: tuple[str, ...],
    length: int,
) -> torch.Tensor:
    out = torch.zeros((int(length),), dtype=torch.float32)
    for name in names:
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(value, dict):
            for key, raw_score in value.items():
                try:
                    idx = int(key)
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < int(length):
                    out[idx] = max(out[idx], score)
            continue
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
        if tensor.numel() != int(length):
            raise ValueError(f"{name} length {tensor.numel()} does not match landmark_weights length {length}")
        out = torch.maximum(out, tensor)
    return out


def _aggregate_per_query_support_vector(
    per_query_support: Mapping[str, Mapping[int, float]],
    *,
    length: int,
) -> torch.Tensor:
    """Collapse real per-query PnP support into a full-Gaussian support prior.

    The selector still consumes the query-level table directly for set utility.
    This vector is only a candidate-pool/static-score bridge so attributed
    sparse-PnP landmarks are not silently dropped when no legacy unary
    ``solver_feedback.pkl`` is provided.
    """

    out = torch.zeros((int(length),), dtype=torch.float32)
    if not isinstance(per_query_support, Mapping):
        return out
    for raw_query_map in per_query_support.values():
        if not isinstance(raw_query_map, Mapping):
            continue
        for raw_gid, raw_value in raw_query_map.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= gid < int(length) and value > 0.0:
                out[gid] += float(value)
    return out


SCORE_COMPONENT_KEYS: dict[str, tuple[str, ...]] = {
    "support": ("landmark_support", "support_score", "solver_support", "hard_gain"),
    "regression_risk": ("candidate_regression_risk", "regression_risk", "native_regression_risk"),
    "dense_worsen_risk": ("dense_worsen_risk", "landmark_dense_worsen_risk"),
    "ambiguity_risk": ("ambiguity_risk", "landmark_ambiguity_risk"),
    "artifact_risk": ("artifact_risk", "render_artifact_risk", "landmark_artifact_risk"),
    "hard_negative_risk": ("hard_negative_risk", "negative_support_risk", "landmark_risk"),
}


def summarize_solver_feedback_score_components(
    feedback_payload: dict[str, Any],
    *,
    support_weight: float = 0.0,
    regression_risk_weight: float = 0.0,
    dense_worsen_risk_weight: float = 0.0,
    ambiguity_risk_weight: float = 0.0,
    artifact_risk_weight: float = 0.0,
    hard_negative_risk_weight: float = 0.0,
) -> dict[str, Any]:
    """Report which requested score/risk components are actually present.

    This prevents over-reading a run as failure-aware when the feedback artifact
    only contains unary landmark support/risk vectors.
    """

    requested_weights = {
        "support": float(support_weight),
        "regression_risk": float(regression_risk_weight),
        "dense_worsen_risk": float(dense_worsen_risk_weight),
        "ambiguity_risk": float(ambiguity_risk_weight),
        "artifact_risk": float(artifact_risk_weight),
        "hard_negative_risk": float(hard_negative_risk_weight),
    }
    components: dict[str, Any] = {}
    missing_weighted: list[str] = []
    for component, keys in SCORE_COMPONENT_KEYS.items():
        present_keys = [key for key in keys if key in feedback_payload]
        weight = requested_weights[component]
        components[component] = {
            "requested_weight": weight,
            "present": bool(present_keys),
            "present_keys": present_keys,
        }
        if weight != 0.0 and not present_keys:
            missing_weighted.append(component)

    return {
        "schema_version": str(feedback_payload.get("schema_version", "unknown")),
        "split_name": str(feedback_payload.get("split_name", feedback_payload.get("split", "unknown"))),
        "components": components,
        "missing_weighted_components": missing_weighted,
        "has_query_level_constraints": any(
            key in feedback_payload for key in ("candidate_gain", "source_loss", "query_coverage", "per_query")
        ),
    }


def build_kc_solver_selection_scores(
    feedback_payload: dict[str, Any],
    *,
    candidate_sampled_idx: torch.Tensor | None = None,
    solver_gain_weight: float = 1.0,
    kc_rank_weight: float = 0.0,
    support_weight: float = 0.0,
    regression_risk_weight: float = 1.0,
    dense_worsen_risk_weight: float = 1.0,
    ambiguity_risk_weight: float = 1.0,
    artifact_risk_weight: float = 1.0,
    hard_negative_risk_weight: float = 1.0,
) -> torch.Tensor:
    """Combine ULF K.C. candidate rank with solver-feedback utility and risks.

    ULF's keypoint-consensus sampler proposes the candidate set/order. This score
    keeps that proposal as a rank prior while allowing train/self-map solver
    traces to boost landmarks that improve PnP support and suppress landmarks
    associated with dense regressions, ambiguity, artifacts, or hard negatives.
    """

    if "landmark_weights" not in feedback_payload:
        raise KeyError("solver feedback artifact must contain landmark_weights")
    solver = torch.as_tensor(feedback_payload["landmark_weights"], dtype=torch.float32).reshape(-1).cpu()
    length = int(solver.numel())
    score = float(solver_gain_weight) * solver

    support = _optional_feedback_vector(
        feedback_payload,
        ("landmark_support", "support_score", "solver_support", "hard_gain"),
        length,
    )
    regression = _optional_feedback_vector(
        feedback_payload,
        ("candidate_regression_risk", "regression_risk", "native_regression_risk"),
        length,
    )
    dense = _optional_feedback_vector(
        feedback_payload,
        ("dense_worsen_risk", "landmark_dense_worsen_risk"),
        length,
    )
    ambiguity = _optional_feedback_vector(
        feedback_payload,
        ("ambiguity_risk", "landmark_ambiguity_risk"),
        length,
    )
    artifact = _optional_feedback_vector(
        feedback_payload,
        ("artifact_risk", "render_artifact_risk", "landmark_artifact_risk"),
        length,
    )
    hard_negative = _optional_feedback_vector(
        feedback_payload,
        ("hard_negative_risk", "negative_support_risk", "landmark_risk"),
        length,
    )

    score = (
        score
        + float(support_weight) * support
        - float(regression_risk_weight) * regression.clamp_min(0.0)
        - float(dense_worsen_risk_weight) * dense.clamp_min(0.0)
        - float(ambiguity_risk_weight) * ambiguity.clamp_min(0.0)
        - float(artifact_risk_weight) * artifact.clamp_min(0.0)
        - float(hard_negative_risk_weight) * hard_negative.clamp_min(0.0)
    )

    if candidate_sampled_idx is not None and float(kc_rank_weight) != 0.0:
        candidate = torch.as_tensor(candidate_sampled_idx, dtype=torch.long).reshape(-1).cpu()
        if candidate.numel() and (int(candidate.min().item()) < 0 or int(candidate.max().item()) >= length):
            raise ValueError("candidate_sampled_idx contains values outside landmark_weights")
        rank_prior = torch.zeros_like(score)
        denom = max(int(candidate.numel()) - 1, 1)
        for rank, gid in enumerate(candidate.tolist()):
            prior = 1.0 - (float(rank) / float(denom))
            rank_prior[int(gid)] = max(rank_prior[int(gid)].item(), prior)
        score = score + float(kc_rank_weight) * rank_prior

    return score.to(torch.float32)


def _resize_mask_channel_cuda(mask_tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    mask = torch.as_tensor(mask_tensor).cuda().float()
    if mask.dim() == 2:
        mask = mask[None, None]
    elif mask.dim() == 3:
        mask = mask[None]
    else:
        raise ValueError(f"unsupported mask shape: {tuple(mask.shape)}")
    if mask.shape[-2:] != (int(height), int(width)):
        mask = F.interpolate(mask, size=(int(height), int(width)), mode="nearest")
    return mask[0, 0].bool()


def accumulate_descriptor_proxy(
    descriptor_sum: torch.Tensor,
    descriptor_count: torch.Tensor,
    *,
    visible_indices: torch.Tensor,
    nearest_keypoint_indices: torch.Tensor,
    near_keypoint: torch.Tensor,
    keypoint_descriptors: torch.Tensor,
) -> None:
    near = torch.as_tensor(near_keypoint, dtype=torch.bool, device=visible_indices.device).reshape(-1)
    if int(near.numel()) == 0 or not bool(near.any()):
        return
    gaussian_ids = torch.as_tensor(visible_indices, dtype=torch.long, device=descriptor_sum.device).reshape(-1)[near]
    keypoint_ids = torch.as_tensor(
        nearest_keypoint_indices,
        dtype=torch.long,
        device=descriptor_sum.device,
    ).reshape(-1)[near]
    if int(gaussian_ids.numel()) == 0:
        return
    descriptors = torch.as_tensor(keypoint_descriptors, dtype=torch.float32, device=descriptor_sum.device)
    descriptors = descriptors[keypoint_ids].reshape(int(gaussian_ids.numel()), -1)
    descriptor_sum.index_add_(0, gaussian_ids, descriptors)
    descriptor_count.index_add_(
        0,
        gaussian_ids,
        torch.ones((int(gaussian_ids.numel()),), dtype=torch.float32, device=descriptor_count.device),
    )


def compute_full_gaussian_kc_visibility_signals(
    *,
    feature_extractor: Any,
    scene_obj: Any,
    gaussians: Any,
    masks: Any,
    config: Mapping[str, Any],
    get_render_visible_mask: Any,
    collect_descriptor_proxy: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute full-Gaussian K.C./visibility/mask signals without sampling.

    This mirrors ULF-Loc's keypoint-consensus projection loop, but returns
    full-length score vectors instead of calling the final KNN sampler.
    """

    import faiss  # type: ignore
    from utils.graphics_utils import fov2focal  # type: ignore

    xyz = gaussians.get_xyz
    count = int(xyz.shape[0])
    view_count = torch.zeros((count,), dtype=torch.float32, device=xyz.device)
    mask_valid_count = torch.zeros((count,), dtype=torch.float32, device=xyz.device)
    vote_sum = torch.zeros((count,), dtype=torch.float32, device=xyz.device)
    descriptor_sum = None
    descriptor_count = None
    if bool(collect_descriptor_proxy):
        descriptor_dim = int(getattr(feature_extractor, "feature_dim", 0))
        if descriptor_dim <= 0:
            raise ValueError("feature_extractor must expose feature_dim when descriptor proxy collection is enabled")
        descriptor_sum = torch.zeros((count, descriptor_dim), dtype=torch.float32, device=xyz.device)
        descriptor_count = torch.zeros((count,), dtype=torch.float32, device=xyz.device)
    index = faiss.IndexFlatL2(2)
    threshold = float(config.get("sample", {}).get("thre_dis", 4.0))
    top_k = int(config.get("sample", {}).get("2D_kpts_num", 2048))

    for viewpoint_cam in tqdm(scene_obj.getTrainCameras(), desc="Full-Gaussian K.C./Visibility Signals"):
        width = int(viewpoint_cam.original_image.shape[2])
        height = int(viewpoint_cam.original_image.shape[1])
        render_visible_mask = get_render_visible_mask(gaussians, viewpoint_cam, width, height)
        view_img = viewpoint_cam.original_image.cuda()

        stable_region = None
        if masks is not None:
            scene_masks = masks[viewpoint_cam.image_name]
            obj_mask = _resize_mask_channel_cuda(scene_masks[0], height, width)
            sky_mask = _resize_mask_channel_cuda(scene_masks[1], height, width)
            distort_mask = _resize_mask_channel_cuda(scene_masks[2], height, width)
            stable_region = obj_mask & distort_mask & sky_mask
            view_img = view_img * (obj_mask & distort_mask)
            view_img[sky_mask.repeat(3, 1, 1) == False] = 0

        extracted = feature_extractor.detectAndCompute(view_img[None], top_k=top_k)[0]
        gt_keypoints = extracted["keypoints"]
        if gt_keypoints.numel() == 0:
            continue
        gt_descriptors = extracted.get("descriptors")
        if bool(collect_descriptor_proxy) and gt_descriptors is None:
            raise KeyError("feature extractor output must include descriptors for descriptor proxy collection")

        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()
        focal_x = fov2focal(viewpoint_cam.FoVx, width)
        focal_y = fov2focal(viewpoint_cam.FoVy, height)
        intrinsics = torch.tensor(
            [[focal_x, 0.0, width / 2], [0.0, focal_y, height / 2], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=xyz.device,
        )
        xyz_homo = torch.cat([xyz, torch.ones(count, 1, device=xyz.device)], dim=-1)
        xyz_cam = (viewmat @ xyz_homo.T)[:3]
        depths = xyz_cam[2]
        positive_depth = depths > 1.0e-6
        xyz_cam_homo = xyz_cam / depths.clamp_min(1.0e-6)
        xy = (intrinsics @ xyz_cam_homo)[:2].long()
        in_image = (
            positive_depth
            & (xy[0] >= 0)
            & (xy[0] < width)
            & (xy[1] >= 0)
            & (xy[1] < height)
        )
        if render_visible_mask is not None:
            visible = in_image & render_visible_mask
        else:
            visible = in_image
        visible_indices = torch.where(visible)[0]
        if visible_indices.numel() == 0:
            continue
        view_count[visible_indices] += 1.0

        xy_visible = xy[:, visible].transpose(1, 0)
        if stable_region is not None:
            stable_at_projection = stable_region[
                xy_visible[:, 1].clamp(0, height - 1).long(),
                xy_visible[:, 0].clamp(0, width - 1).long(),
            ].bool()
            mask_valid_count[visible_indices[stable_at_projection]] += 1.0
        else:
            mask_valid_count[visible_indices] += 1.0

        index.reset()
        index.add(gt_keypoints.detach().cpu().numpy().astype("float32"))
        dists, nearest = index.search(xy_visible.detach().cpu().numpy().astype("float32"), k=1)
        near_keypoint = torch.from_numpy((dists <= threshold).reshape(-1)).to(device=xyz.device)
        vote_sum[visible_indices[near_keypoint]] += 1.0
        if descriptor_sum is not None and descriptor_count is not None and gt_descriptors is not None:
            accumulate_descriptor_proxy(
                descriptor_sum,
                descriptor_count,
                visible_indices=visible_indices,
                nearest_keypoint_indices=torch.from_numpy(nearest.reshape(-1)).to(device=xyz.device),
                near_keypoint=near_keypoint,
                keypoint_descriptors=gt_descriptors.to(device=xyz.device),
            )

    mask_validity = torch.zeros_like(mask_valid_count)
    observed = view_count > 0
    mask_validity[observed] = mask_valid_count[observed] / view_count[observed].clamp_min(1.0)
    result = {
        "kc_score": vote_sum.detach().cpu(),
        "visibility_score": view_count.detach().cpu(),
        "mask_validity": mask_validity.detach().cpu(),
    }
    if descriptor_sum is not None and descriptor_count is not None:
        descriptor_proxy = torch.zeros_like(descriptor_sum)
        has_descriptor = descriptor_count > 0.0
        descriptor_proxy[has_descriptor] = descriptor_sum[has_descriptor] / descriptor_count[has_descriptor, None]
        descriptor_proxy = F.normalize(descriptor_proxy, p=2, dim=1)
        result["descriptor_proxy"] = descriptor_proxy.detach().cpu()
        result["descriptor_proxy_count"] = descriptor_count.detach().cpu()
    return result


def _score_for_gid(scores: torch.Tensor | None, gid: int, default: float = 1.0) -> float:
    if scores is None:
        return float(default)
    if 0 <= int(gid) < int(scores.numel()):
        return float(scores[int(gid)].item())
    return float(default)


def choose_native_safe_core_and_tail(
    native_sampled_idx: torch.Tensor,
    *,
    safe_count: int,
    protection_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split native landmarks into a score-protected core and a droppable tail.

    ULF's sampled_idx is commonly sorted by Gaussian id after torch.unique, so
    using the first N entries as a "safe" core protects id order rather than
    localization utility. Ties keep native order for deterministic behavior.
    """

    native = torch.as_tensor(native_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    count = int(native.numel())
    keep = min(max(0, int(safe_count)), count)
    if keep == count:
        return native.clone(), torch.empty((0,), dtype=torch.long)
    if keep == 0:
        return torch.empty((0,), dtype=torch.long), native.clone()
    scores = None
    if protection_scores is not None:
        scores = torch.as_tensor(protection_scores, dtype=torch.float32).reshape(-1).cpu()
    ranked = sorted(
        enumerate(native.tolist()),
        key=lambda item: (-_score_for_gid(scores, int(item[1])), int(item[0])),
    )
    safe_ranks = {int(rank) for rank, _ in ranked[:keep]}
    safe = [int(gid) for rank, gid in enumerate(native.tolist()) if rank in safe_ranks]
    tail = [int(gid) for rank, gid in enumerate(native.tolist()) if rank not in safe_ranks]
    return torch.tensor(safe, dtype=torch.long), torch.tensor(tail, dtype=torch.long)


def _tail_keep_order(native_tail: torch.Tensor, scores: torch.Tensor | None) -> list[int]:
    tail = torch.as_tensor(native_tail, dtype=torch.long).reshape(-1).cpu().tolist()
    return [
        int(gid)
        for _, gid in sorted(
            enumerate(tail),
            key=lambda item: (-_score_for_gid(scores, int(item[1])), int(item[0])),
        )
    ]


def merge_native_safe_core_sampled_idx(
    native_sampled_idx: torch.Tensor,
    candidate_sampled_idx: torch.Tensor,
    *,
    solver_weights: torch.Tensor | None = None,
    safe_core_fraction: float = 0.9,
    max_solver_insertions: int | None = None,
    min_solver_insertion_score: float | None = None,
    min_solver_insertion_margin: float = 0.0,
) -> torch.Tensor:
    native = torch.as_tensor(native_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    candidate = torch.as_tensor(candidate_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    if native.numel() == 0 or candidate.numel() == 0:
        raise ValueError("native and candidate sampled_idx must be non-empty")
    if torch.unique(native).numel() != native.numel():
        raise ValueError("native sampled_idx must be unique")
    if torch.unique(candidate).numel() != candidate.numel():
        raise ValueError("candidate sampled_idx must be unique")

    target_count = int(candidate.numel())
    safe_fraction = min(1.0, max(0.0, float(safe_core_fraction)))
    safe_count = min(int(round(target_count * safe_fraction)), int(native.numel()), target_count)
    insertion_budget = target_count - safe_count
    if max_solver_insertions is not None:
        insertion_budget = min(insertion_budget, max(0, int(max_solver_insertions)))
    if insertion_budget <= 0 and int(native.numel()) == target_count:
        return native.clone()

    cached_weights = (
        torch.as_tensor(solver_weights, dtype=torch.float32).reshape(-1).cpu()
        if solver_weights is not None
        else None
    )
    safe_core, native_tail_t = choose_native_safe_core_and_tail(
        native,
        safe_count=safe_count,
        protection_scores=cached_weights,
    )

    selected: list[int] = []
    seen: set[int] = set()
    for value in safe_core.tolist():
        idx = int(value)
        selected.append(idx)
        seen.add(idx)

    candidate_pool = [int(value) for value in candidate.tolist() if int(value) not in seen]
    if solver_weights is not None and candidate_pool:
        weights = torch.as_tensor(solver_weights, dtype=torch.float32).reshape(-1).cpu()

        def sort_key(item: tuple[int, int]) -> tuple[float, int]:
            rank, gid = item
            weight = float(weights[gid].item()) if 0 <= gid < int(weights.numel()) else 1.0
            return (-weight, rank)

        candidate_pool = [gid for _, gid in sorted(enumerate(candidate_pool), key=sort_key)]

    native_tail = [int(value) for value in native_tail_t.tolist()]

    def _weight(gid: int) -> float:
        if cached_weights is None:
            return 1.0
        return float(cached_weights[gid].item()) if 0 <= gid < int(cached_weights.numel()) else 0.0

    native_fallback_score = min((_weight(gid) for gid in native_tail), default=0.0)
    min_score = None if min_solver_insertion_score is None else float(min_solver_insertion_score)
    min_margin = float(min_solver_insertion_margin)

    inserted_count = 0
    for gid in candidate_pool:
        if len(selected) >= target_count:
            break
        if len(selected) - safe_count >= insertion_budget:
            break
        candidate_score = _weight(gid)
        if min_score is not None and candidate_score < min_score:
            continue
        if solver_weights is not None and min_margin > 0.0:
            if candidate_score < native_fallback_score + min_margin:
                continue
        selected.append(gid)
        seen.add(gid)
        inserted_count += 1

    if inserted_count == 0 and int(native.numel()) == target_count:
        return native.clone()

    for value in _tail_keep_order(native_tail_t, cached_weights):
        if len(selected) >= target_count:
            break
        idx = int(value)
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)

    if len(selected) != target_count:
        raise ValueError(f"merged sampled_idx has {len(selected)} entries, expected {target_count}")
    return torch.tensor(selected, dtype=torch.long)


def merge_solver_coverage_coreset_sampled_idx(
    native_sampled_idx: torch.Tensor,
    candidate_sampled_idx: torch.Tensor,
    *,
    solver_scores: torch.Tensor,
    cluster_ids: torch.Tensor | None = None,
    safe_core_fraction: float = 0.9,
    max_solver_insertions: int | None = None,
    max_insertions_per_cluster: int = 1,
    min_solver_insertion_score: float | None = None,
    min_solver_insertion_margin: float = 0.0,
) -> torch.Tensor:
    """Native-safe greedy coreset with a simple spatial/cluster saturation guard.

    The existing native-safe merge is a unary ranker. This variant keeps the
    same paper-safe fixed-budget behavior but prevents inserted solver-feedback
    candidates from collapsing into a single local/repeated structure cluster.
    Native tail points remain the fallback, so the output size and evaluator
    path are unchanged.
    """

    native = torch.as_tensor(native_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    candidate = torch.as_tensor(candidate_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    scores = torch.as_tensor(solver_scores, dtype=torch.float32).reshape(-1).cpu()
    if native.numel() == 0 or candidate.numel() == 0:
        raise ValueError("native and candidate sampled_idx must be non-empty")
    if torch.unique(native).numel() != native.numel():
        raise ValueError("native sampled_idx must be unique")
    if torch.unique(candidate).numel() != candidate.numel():
        raise ValueError("candidate sampled_idx must be unique")
    if candidate.numel() and (int(candidate.min().item()) < 0 or int(candidate.max().item()) >= int(scores.numel())):
        raise ValueError("candidate_sampled_idx contains values outside solver_scores")

    clusters = None
    if cluster_ids is not None:
        clusters = torch.as_tensor(cluster_ids, dtype=torch.long).reshape(-1).cpu()
        if clusters.numel() != scores.numel():
            raise ValueError("cluster_ids length must match solver_scores")

    target_count = int(candidate.numel())
    safe_fraction = min(1.0, max(0.0, float(safe_core_fraction)))
    safe_count = min(int(round(target_count * safe_fraction)), int(native.numel()), target_count)
    insertion_budget = target_count - safe_count
    if max_solver_insertions is not None:
        insertion_budget = min(insertion_budget, max(0, int(max_solver_insertions)))
    if insertion_budget <= 0 and int(native.numel()) == target_count:
        return native.clone()

    safe_core, native_tail_t = choose_native_safe_core_and_tail(
        native,
        safe_count=safe_count,
        protection_scores=scores,
    )

    selected: list[int] = []
    seen: set[int] = set()
    for value in safe_core.tolist():
        idx = int(value)
        selected.append(idx)
        seen.add(idx)

    native_tail = [int(value) for value in native_tail_t.tolist()]
    native_fallback_score = (
        min(float(scores[gid].item()) for gid in native_tail if 0 <= gid < int(scores.numel()))
        if native_tail
        else 0.0
    )
    min_score = None if min_solver_insertion_score is None else float(min_solver_insertion_score)
    min_margin = float(min_solver_insertion_margin)
    max_per_cluster = max(1, int(max_insertions_per_cluster))
    inserted_cluster_counts: dict[int, int] = {}

    candidate_pool = [int(value) for value in candidate.tolist() if int(value) not in seen]
    candidate_pool.sort(key=lambda gid: float(scores[gid].item()) if 0 <= gid < int(scores.numel()) else -float("inf"), reverse=True)

    inserted_count = 0
    for gid in candidate_pool:
        if len(selected) >= target_count:
            break
        if len(selected) - safe_count >= insertion_budget:
            break
        candidate_score = float(scores[gid].item()) if 0 <= gid < int(scores.numel()) else 0.0
        if min_score is not None and candidate_score < min_score:
            continue
        if min_margin > 0.0 and candidate_score < native_fallback_score + min_margin:
            continue
        cluster = int(clusters[gid].item()) if clusters is not None else gid
        if inserted_cluster_counts.get(cluster, 0) >= max_per_cluster:
            continue
        selected.append(gid)
        seen.add(gid)
        inserted_cluster_counts[cluster] = inserted_cluster_counts.get(cluster, 0) + 1
        inserted_count += 1

    if inserted_count == 0 and int(native.numel()) == target_count:
        return native.clone()

    for value in _tail_keep_order(native_tail_t, scores):
        if len(selected) >= target_count:
            break
        idx = int(value)
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)

    if len(selected) != target_count:
        raise ValueError(f"merged sampled_idx has {len(selected)} entries, expected {target_count}")
    return torch.tensor(selected, dtype=torch.long)


@dataclass(frozen=True)
class QueryCoverageMergeResult:
    sampled_idx: torch.Tensor
    metadata: dict[str, Any]


def merge_solver_query_coverage_sampled_idx(
    native_sampled_idx: torch.Tensor,
    candidate_sampled_idx: torch.Tensor,
    *,
    solver_scores: torch.Tensor,
    solver_constraints: dict[str, Any],
    safe_core_fraction: float = 0.9,
    max_solver_insertions: int | None = None,
    max_drop_scan: int = 1,
    min_coverage_gain: float = 0.0,
    min_query_coverage: float = 1.0,
    coverage_saturation_mode: str = "none",
    coverage_saturation_percentile: float = 0.75,
    coverage_saturation_fraction: float = 1.0,
    easy_query_gain_decay: float = 0.0,
    tail_cvar_alpha: float = 0.0,
    tail_query_gain_boost: float = 1.0,
    hard_query_min_gain: float = 0.0,
    saturation_drop_protection_weight: float = 0.0,
    require_candidate_gain: bool = False,
) -> QueryCoverageMergeResult:
    """Native-safe ULF merge driven by hard-query solver coverage constraints."""

    native = torch.as_tensor(native_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    candidate = torch.as_tensor(candidate_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    scores = torch.as_tensor(solver_scores, dtype=torch.float32).reshape(-1).cpu()
    if native.numel() == 0 or candidate.numel() == 0:
        raise ValueError("native and candidate sampled_idx must be non-empty")
    if torch.unique(native).numel() != native.numel():
        raise ValueError("native sampled_idx must be unique")
    if torch.unique(candidate).numel() != candidate.numel():
        raise ValueError("candidate sampled_idx must be unique")
    if candidate.numel() and (int(candidate.min().item()) < 0 or int(candidate.max().item()) >= int(scores.numel())):
        raise ValueError("candidate_sampled_idx contains values outside solver_scores")

    target_count = int(candidate.numel())
    safe_fraction = min(1.0, max(0.0, float(safe_core_fraction)))
    safe_count = min(int(round(target_count * safe_fraction)), int(native.numel()), target_count)
    insertion_budget = target_count - safe_count
    if max_solver_insertions is not None:
        insertion_budget = min(insertion_budget, max(0, int(max_solver_insertions)))

    evidence_mask = torch.zeros_like(scores, dtype=torch.bool)
    if candidate.numel():
        evidence_mask[candidate] = True
    tables = build_solver_coverage_tables(solver_constraints)
    thresholds = dict(solver_constraints.get("thresholds", {}))
    checker = None
    if require_candidate_gain:
        checker = make_replacement_admissibility_checker(
            hard_query_ids=tables.hard_query_ids,
            candidate_gain=tables.candidate_gain,
            source_loss=tables.source_loss,
            min_support_delta=float(thresholds.get("min_support_delta", 0.0)),
            min_viable_tuple_delta=float(thresholds.get("min_viable_tuple_delta", 0.0)),
            min_logdet_delta=float(thresholds.get("min_logdet_delta", 0.0)),
            min_min_eigen_delta=float(thresholds.get("min_min_eigen_delta", 0.0)),
            max_dense_worsen_delta=float(thresholds.get("max_dense_worsen_delta", 0.0)),
            max_ambiguity_delta=float(thresholds.get("max_ambiguity_delta", 0.0)),
            cvar_alpha=thresholds.get("cvar_alpha"),
            min_cvar_score=float(thresholds.get("min_cvar_score", 0.0)),
            cvar_weights=thresholds.get("cvar_weights"),
            require_candidate_gain=True,
        )

    safe_core, _ = choose_native_safe_core_and_tail(
        native,
        safe_count=safe_count,
        protection_scores=scores,
    )

    result = solver_coverage_local_edit(
        source_idx=native,
        candidate_pool=candidate,
        utility=scores,
        evidence_mask=evidence_mask,
        coverage_tables=tables,
        safe_core=safe_core,
        is_admissible=(lambda add_id, drop_id: bool(checker(add_id=add_id, drop_id=drop_id).admissible))
        if checker is not None
        else None,
        max_edits=int(insertion_budget),
        max_drop_scan=int(max_drop_scan),
        min_coverage_gain=float(min_coverage_gain),
        min_query_coverage=float(min_query_coverage),
        coverage_saturation_mode=str(coverage_saturation_mode),
        coverage_saturation_percentile=float(coverage_saturation_percentile),
        coverage_saturation_fraction=float(coverage_saturation_fraction),
        easy_query_gain_decay=float(easy_query_gain_decay),
        tail_cvar_alpha=float(tail_cvar_alpha),
        tail_query_gain_boost=float(tail_query_gain_boost),
        hard_query_min_gain=float(hard_query_min_gain),
        saturation_drop_protection_weight=float(saturation_drop_protection_weight),
    )
    sampled = torch.as_tensor(result["sampled_idx"], dtype=torch.long).reshape(-1).cpu()
    if int(sampled.numel()) != int(native.numel()):
        raise ValueError(f"query coverage merge produced {sampled.numel()} entries, expected {native.numel()}")
    return QueryCoverageMergeResult(sampled_idx=sampled, metadata=dict(result["metadata"]))


def _coverage_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    q = min(1.0, max(0.0, float(percentile)))
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * float(len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - float(lo)
    return float((1.0 - frac) * ordered[lo] + frac * ordered[hi])


def _query_metric_sum(query_map: dict[str, float], hard_query_ids: tuple[str, ...]) -> float:
    return float(sum(max(0.0, float(query_map.get(str(query_id), 0.0))) for query_id in hard_query_ids))


def _static_sparse_query_score(
    *,
    gid: int,
    source_set: set[int],
    tables: Any,
    initial_coverage: dict[str, float],
    saturation_targets: dict[str, float] | None,
    hard_query_weight: float,
    source_protection_weight: float,
    easy_query_gain_decay: float,
) -> tuple[float, dict[str, float]]:
    if int(gid) in source_set:
        query_map = tables.source_utility.get(int(gid), {})
        source_bonus = float(source_protection_weight) * _query_metric_sum(query_map, tables.hard_query_ids)
    else:
        query_map = tables.candidate_utility.get(int(gid), {})
        source_bonus = 0.0

    per_query_effective: dict[str, float] = {}
    query_score = 0.0
    for query_id in tables.hard_query_ids:
        raw = max(0.0, float(query_map.get(str(query_id), 0.0)))
        if raw <= 0.0:
            per_query_effective[str(query_id)] = 0.0
            continue
        if saturation_targets is None:
            effective = raw / (1.0 + max(0.0, float(initial_coverage.get(str(query_id), 0.0))) ** 0.5)
        else:
            target = float(saturation_targets.get(str(query_id), 0.0))
            remaining = max(0.0, target - max(0.0, float(initial_coverage.get(str(query_id), 0.0))))
            capped = min(raw, remaining)
            effective = capped + max(0.0, float(easy_query_gain_decay)) * max(0.0, raw - capped)
        per_query_effective[str(query_id)] = float(effective)
        query_score += float(effective)
    return float(hard_query_weight) * query_score + source_bonus, per_query_effective


def select_sparse_solver_aware_kc_sampled_idx(
    native_sampled_idx: torch.Tensor,
    candidate_sampled_idx: torch.Tensor,
    *,
    solver_scores: torch.Tensor,
    solver_constraints: dict[str, Any],
    target_count: int | None = None,
    cluster_ids: torch.Tensor | None = None,
    max_per_cluster: int = 0,
    solver_score_weight: float = 0.1,
    hard_query_weight: float = 2.0,
    source_protection_weight: float = 1.0,
    coverage_saturation_mode: str = "native_percentile",
    coverage_saturation_percentile: float = 0.75,
    coverage_saturation_fraction: float = 1.0,
    easy_query_gain_decay: float = 0.25,
) -> QueryCoverageMergeResult:
    """Select a full ULF K.C. landmark set with sparse PnP solver feedback.

    This is not a native+M-insertions edit. The selected set is ranked from the
    union of native ULF landmarks and the current K.C. candidate proposal. Native
    landmarks are protected only when sparse train/self-map inlier coverage says
    they explain hard queries.
    """

    native = torch.as_tensor(native_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    candidate = torch.as_tensor(candidate_sampled_idx, dtype=torch.long).reshape(-1).cpu()
    scores = torch.as_tensor(solver_scores, dtype=torch.float32).reshape(-1).cpu()
    if native.numel() == 0 or candidate.numel() == 0:
        raise ValueError("native and candidate sampled_idx must be non-empty")
    if torch.unique(native).numel() != native.numel():
        raise ValueError("native sampled_idx must be unique")
    if torch.unique(candidate).numel() != candidate.numel():
        raise ValueError("candidate sampled_idx must be unique")
    if candidate.numel() and (int(candidate.min().item()) < 0 or int(candidate.max().item()) >= int(scores.numel())):
        raise ValueError("candidate_sampled_idx contains values outside solver_scores")
    if native.numel() and (int(native.min().item()) < 0 or int(native.max().item()) >= int(scores.numel())):
        raise ValueError("native_sampled_idx contains values outside solver_scores")

    target = int(candidate.numel() if target_count is None else target_count)
    if target <= 0:
        raise ValueError("target_count must be positive")
    source_set = {int(value) for value in native.tolist()}
    tables = build_solver_coverage_tables(solver_constraints)
    initial_coverage = {
        str(query_id): 0.0
        for query_id in tables.hard_query_ids
    }
    for gid in source_set:
        for query_id, value in tables.source_utility.get(int(gid), {}).items():
            if query_id in initial_coverage:
                initial_coverage[query_id] += max(0.0, float(value))

    saturation_mode = str(coverage_saturation_mode or "none")
    saturation_targets: dict[str, float] | None = None
    if saturation_mode == "native_percentile":
        target_value = _coverage_percentile(list(initial_coverage.values()), float(coverage_saturation_percentile))
        saturation_targets = {query_id: float(target_value) for query_id in initial_coverage}
    elif saturation_mode == "native_fraction":
        saturation_targets = {
            query_id: max(0.0, float(value) * max(0.0, float(coverage_saturation_fraction)))
            for query_id, value in initial_coverage.items()
        }
    elif saturation_mode != "none":
        raise ValueError("coverage_saturation_mode must be one of: none, native_percentile, native_fraction")

    clusters = None
    if cluster_ids is not None:
        clusters = torch.as_tensor(cluster_ids, dtype=torch.long).reshape(-1).cpu()
        if clusters.numel() != scores.numel():
            raise ValueError("cluster_ids length must match solver_scores")
    max_cluster = max(0, int(max_per_cluster))

    universe: list[int] = []
    seen: set[int] = set()
    for values in (candidate.tolist(), native.tolist()):
        for value in values:
            gid = int(value)
            if gid in seen:
                continue
            seen.add(gid)
            universe.append(gid)
    if len(universe) < target:
        raise ValueError(f"candidate/native union has {len(universe)} landmarks, below target_count {target}")

    ranked: list[tuple[float, int, dict[str, float]]] = []
    for gid in universe:
        query_score, per_query = _static_sparse_query_score(
            gid=int(gid),
            source_set=source_set,
            tables=tables,
            initial_coverage=initial_coverage,
            saturation_targets=saturation_targets,
            hard_query_weight=float(hard_query_weight),
            source_protection_weight=float(source_protection_weight),
            easy_query_gain_decay=float(easy_query_gain_decay),
        )
        base_score = float(scores[int(gid)].item()) if 0 <= int(gid) < int(scores.numel()) else 0.0
        total = float(solver_score_weight) * base_score + float(query_score)
        ranked.append((float(total), int(gid), per_query))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected: list[int] = []
    selected_set: set[int] = set()
    cluster_counts: dict[int, int] = {}
    cluster_skipped = 0

    def _try_add(gid: int, *, enforce_cluster: bool) -> bool:
        if gid in selected_set:
            return False
        if enforce_cluster and clusters is not None and max_cluster > 0:
            cluster = int(clusters[int(gid)].item())
            if cluster_counts.get(cluster, 0) >= max_cluster:
                return False
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        elif clusters is not None and max_cluster > 0:
            cluster = int(clusters[int(gid)].item())
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        selected.append(int(gid))
        selected_set.add(int(gid))
        return True

    for _, gid, _ in ranked:
        if len(selected) >= target:
            break
        before = len(selected)
        _try_add(int(gid), enforce_cluster=True)
        if len(selected) == before:
            cluster_skipped += 1
    if len(selected) < target:
        for _, gid, _ in ranked:
            if len(selected) >= target:
                break
            _try_add(int(gid), enforce_cluster=False)

    sampled = torch.tensor(selected[:target], dtype=torch.long)
    selected_native = source_set & set(sampled.tolist())
    final_coverage = {
        str(query_id): 0.0
        for query_id in tables.hard_query_ids
    }
    for gid in sampled.tolist():
        table = tables.source_utility if int(gid) in source_set else tables.candidate_utility
        for query_id, value in table.get(int(gid), {}).items():
            if query_id in final_coverage:
                final_coverage[query_id] += max(0.0, float(value))

    metadata = {
        "selection_policy": "sparse_solver_aware_kc",
        "budget_mode": "full_set",
        "source_sampled_count": int(native.numel()),
        "candidate_pool_count": int(candidate.numel()),
        "output_sampled_count": int(sampled.numel()),
        "native_kept_count": int(len(selected_native)),
        "native_dropped_count": int(len(source_set - selected_native)),
        "added_non_native_count": int(len(set(sampled.tolist()) - source_set)),
        "hard_query_count": int(len(tables.hard_query_ids)),
        "initial_query_coverage": initial_coverage,
        "final_query_coverage": final_coverage,
        "solver_score_weight": float(solver_score_weight),
        "hard_query_weight": float(hard_query_weight),
        "source_protection_weight": float(source_protection_weight),
        "coverage_saturation": {
            "enabled": saturation_targets is not None,
            "mode": saturation_mode,
            "percentile": float(coverage_saturation_percentile),
            "fraction": float(coverage_saturation_fraction),
            "target_min": float(min(saturation_targets.values())) if saturation_targets else 0.0,
            "target_max": float(max(saturation_targets.values())) if saturation_targets else 0.0,
        },
        "cluster_cap": {
            "enabled": bool(clusters is not None and max_cluster > 0),
            "max_per_cluster": int(max_cluster),
            "skipped_first_pass": int(cluster_skipped),
            "unique_selected_clusters": int(len({int(clusters[int(gid)].item()) for gid in sampled.tolist()}))
            if clusters is not None
            else 0,
        },
    }
    return QueryCoverageMergeResult(sampled_idx=sampled, metadata=metadata)


def build_spatial_cluster_ids(xyz: torch.Tensor, *, grid_size_m: float) -> torch.Tensor:
    points = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3).cpu()
    if points.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    grid = float(grid_size_m)
    if grid <= 0.0:
        return torch.arange(points.shape[0], dtype=torch.long)
    cells = torch.floor(points / grid).to(torch.long)
    # A compact inverse from torch.unique(dim=0) is unnecessarily expensive for
    # million-scale 3DGS maps; the saturation guard only needs equal grid cells
    # to share a deterministic id.
    primes = torch.tensor([73856093, 19349663, 83492791], dtype=torch.long)
    return (cells * primes).sum(dim=1).to(torch.long)


def _git_status_at(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(path), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resample ULF-Loc landmarks/features on a fixed native map using solver feedback."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--source_model_path", required=True, type=Path)
    parser.add_argument("--output_model_path", required=True, type=Path)
    parser.add_argument("--solver_feedback", default=None, type=Path)
    parser.add_argument("--solver_sampling_alpha", default=None, type=float)
    parser.add_argument("--solver_fusion_alpha", default=None, type=float)
    parser.add_argument(
        "--reuse_sampled_idx",
        default=None,
        type=Path,
        help=(
            "Optional sampled_idx artifact to preload into the output eval directory. "
            "This skips ULF-Loc K.C. sampling and isolates descriptor-fusion ablations."
        ),
    )
    parser.add_argument(
        "--ray_solver_feedback",
        default=None,
        type=Path,
        help=(
            "Optional ray-attributed full-Gaussian solver feedback artifact. "
            "Used only during map export/set selection, never during per-query eval routing."
        ),
    )
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--train_list", default=None, type=Path)
    parser.add_argument("--sample_random_seed", default=None, type=int)
    parser.add_argument("--native_sampled_idx", default=None, type=Path)
    parser.add_argument("--native_safe_core_fraction", default=1.0, type=float)
    parser.add_argument("--max_solver_insertions", default=None, type=int)
    parser.add_argument("--min_solver_insertion_score", default=None, type=float)
    parser.add_argument("--min_solver_insertion_margin", default=0.0, type=float)
    parser.add_argument(
        "--selection_policy",
        default="solver_weight",
        choices=(
            "solver_weight",
            "kc_solver",
            "solver_coverage",
            "solver_query_coverage",
            "sparse_solver_aware_kc",
            "full_gaussian_sparse_set",
        ),
        help="Ranking policy for native-safe insertions after ULF K.C. sampling.",
    )
    parser.add_argument("--kc_rank_weight", default=0.0, type=float)
    parser.add_argument("--solver_gain_weight", default=1.0, type=float)
    parser.add_argument("--support_weight", default=0.0, type=float)
    parser.add_argument("--regression_risk_weight", default=1.0, type=float)
    parser.add_argument("--dense_worsen_risk_weight", default=1.0, type=float)
    parser.add_argument("--ambiguity_risk_weight", default=1.0, type=float)
    parser.add_argument("--artifact_risk_weight", default=1.0, type=float)
    parser.add_argument("--hard_negative_risk_weight", default=1.0, type=float)
    parser.add_argument("--coverage_grid_size_m", default=0.5, type=float)
    parser.add_argument("--max_insertions_per_cluster", default=1, type=int)
    parser.add_argument("--solver_admissibility_path", default=None, type=Path)
    parser.add_argument("--query_coverage_max_drop_scan", default=1, type=int)
    parser.add_argument("--query_coverage_min_coverage_gain", default=0.0, type=float)
    parser.add_argument("--query_coverage_min_query_coverage", default=1.0, type=float)
    parser.add_argument(
        "--coverage_saturation_mode",
        default="none",
        choices=("none", "native_percentile", "native_fraction"),
    )
    parser.add_argument("--coverage_saturation_percentile", default=0.75, type=float)
    parser.add_argument("--coverage_saturation_fraction", default=1.0, type=float)
    parser.add_argument("--easy_query_gain_decay", default=0.0, type=float)
    parser.add_argument("--tail_cvar_alpha", default=0.0, type=float)
    parser.add_argument("--tail_query_gain_boost", default=1.0, type=float)
    parser.add_argument("--hard_query_min_gain", default=0.0, type=float)
    parser.add_argument("--saturation_drop_protection_weight", default=0.0, type=float)
    parser.add_argument("--sparse_solver_score_weight", default=0.1, type=float)
    parser.add_argument("--sparse_hard_query_weight", default=2.0, type=float)
    parser.add_argument("--sparse_source_protection_weight", default=1.0, type=float)
    parser.add_argument("--full_set_max_landmarks", default=0, type=int)
    parser.add_argument("--full_set_min_landmarks", default=0, type=int)
    parser.add_argument("--full_set_min_marginal_gain", default=0.05, type=float)
    parser.add_argument("--full_set_candidate_pool_size", default=80000, type=int)
    parser.add_argument("--full_set_candidate_pool_kc_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_visibility_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_mask_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_solver_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_match_strength_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_query_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_query_cell_top_k", default=0, type=int)
    parser.add_argument("--full_set_candidate_pool_query_depth_top_k", default=0, type=int)
    parser.add_argument("--full_set_source_anchor_idx", default=None, type=Path)
    parser.add_argument("--full_set_target_query_support", default=64.0, type=float)
    parser.add_argument("--full_set_min_visibility", default=1.0, type=float)
    parser.add_argument("--full_set_min_mask_validity", default=0.5, type=float)
    parser.add_argument("--full_set_max_per_spatial_cluster", default=0, type=int)
    parser.add_argument("--full_set_visibility_weight", default=0.5, type=float)
    parser.add_argument("--full_set_mask_weight", default=0.5, type=float)
    parser.add_argument("--full_set_solver_weight", default=1.0, type=float)
    parser.add_argument("--full_set_query_support_weight", default=2.0, type=float)
    parser.add_argument("--full_set_geometry_weight", default=0.5, type=float)
    parser.add_argument("--full_set_depth_weight", default=0.25, type=float)
    parser.add_argument("--full_set_query_geometry_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_depth_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_bearing_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_pnp_geometry_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_pnp_distance_scale_m", default=2.0, type=float)
    parser.add_argument("--full_set_support_match_strength", default=None, type=Path)
    parser.add_argument("--full_set_match_competition_risk", default=None, type=Path)
    parser.add_argument("--full_set_support_match_strength_weight", default=0.0, type=float)
    parser.add_argument("--full_set_match_competition_risk_weight", default=0.0, type=float)
    parser.add_argument("--full_set_match_competition_risk_reject_threshold", default=0.0, type=float)
    parser.add_argument("--full_set_query_match_dominance", default=None, type=Path)
    parser.add_argument("--full_set_query_match_strength_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_match_competition_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_match_competition_ratio_weight", default=0.0, type=float)
    parser.add_argument("--full_set_candidate_pool_query_match_strength_top_k", default=0, type=int)
    parser.add_argument("--full_set_query_match_reference_sampled_idx", default=None, type=Path)
    parser.add_argument("--full_set_query_match_risk_reference_margin", default=0.0, type=float)
    parser.add_argument("--full_set_query_match_risk_reference_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_match_nonreference_competition_weight", default=0.0, type=float)
    parser.add_argument("--full_set_query_prefill_target_fraction", default=0.0, type=float)
    parser.add_argument("--full_set_query_prefill_max_candidates_per_query", default=0, type=int)
    parser.add_argument("--full_set_query_prefill_max_per_spatial_cluster", default=0, type=int)
    parser.add_argument("--full_set_query_prefill_min_cells_per_query", default=0, type=int)
    parser.add_argument("--full_set_query_prefill_min_depth_bins_per_query", default=0, type=int)
    parser.add_argument("--full_set_query_prefill_hard_coverage_bonus", default=1.0e6, type=float)
    parser.add_argument("--full_set_source_query_prefill_target_fraction", default=0.0, type=float)
    parser.add_argument("--full_set_source_query_prefill_max_candidates_per_query", default=0, type=int)
    parser.add_argument("--full_set_source_query_prefill_max_per_spatial_cluster", default=0, type=int)
    parser.add_argument("--full_set_source_query_prefill_min_cells_per_query", default=0, type=int)
    parser.add_argument("--full_set_source_query_prefill_min_depth_bins_per_query", default=0, type=int)
    parser.add_argument("--full_set_validation_query_prefill_target_fraction", default=0.0, type=float)
    parser.add_argument("--full_set_validation_query_prefill_max_candidates_per_query", default=0, type=int)
    parser.add_argument("--full_set_validation_query_prefill_max_per_spatial_cluster", default=0, type=int)
    parser.add_argument("--full_set_validation_query_prefill_min_cells_per_query", default=0, type=int)
    parser.add_argument("--full_set_validation_query_prefill_min_depth_bins_per_query", default=0, type=int)
    parser.add_argument("--full_set_validation_query_prefill_force_all", action="store_true")
    parser.add_argument("--full_set_main_min_query_gain", default=0.0, type=float)
    parser.add_argument("--full_set_main_dynamic_lookahead", default=0, type=int)
    parser.add_argument("--full_set_kc_anchor_count", default=0, type=int)
    parser.add_argument("--full_set_kc_anchor_min_score", default=0.0, type=float)
    parser.add_argument("--full_set_kc_anchor_max_per_spatial_cluster", default=0, type=int)
    parser.add_argument("--full_set_source_anchor_mode", default="hard", choices=("hard", "soft", "off"))
    parser.add_argument("--full_set_source_anchor_weight", default=0.0, type=float)
    parser.add_argument("--full_set_force_source_anchor", action="store_true")
    parser.add_argument("--full_set_descriptor_conflict_graph", default=None, type=Path)
    parser.add_argument("--full_set_descriptor_conflict_weight", default=0.0, type=float)
    parser.add_argument("--full_set_auto_descriptor_conflict_top_k", default=0, type=int)
    parser.add_argument("--full_set_auto_descriptor_conflict_candidate_limit", default=8192, type=int)
    parser.add_argument("--full_set_auto_descriptor_conflict_min_cosine", default=0.95, type=float)
    parser.add_argument("--full_set_auto_descriptor_conflict_min_spatial_distance_m", default=1.0, type=float)
    parser.add_argument("--full_set_sparse_validation_profile", default=None, type=Path)
    parser.add_argument(
        "--full_set_allow_metric_only_sparse_validation_profile",
        action="store_true",
        help="Allow diagnostic SparseSet runs with metric-only sparse validation profiles.",
    )
    parser.add_argument(
        "--full_set_sparse_validation_support_scope",
        default="all",
        choices=("all", "protected_regressions"),
    )
    parser.add_argument("--full_set_sparse_validation_risk_weight", default=0.0, type=float)
    parser.add_argument("--full_set_sparse_validation_risk_reject_threshold", default=0.0, type=float)
    parser.add_argument("--full_set_sparse_validation_protected_hard_reject_exempt", action="store_true")
    parser.add_argument("--full_set_sparse_validation_source_anchor_risk_exempt", action="store_true")
    parser.add_argument("--full_set_sparse_validation_risk_expand_top_k", default=0, type=int)
    parser.add_argument("--full_set_sparse_validation_risk_expand_neighbors_per_seed", default=16, type=int)
    parser.add_argument("--full_set_sparse_validation_risk_expand_candidate_limit", default=65536, type=int)
    parser.add_argument("--full_set_sparse_validation_risk_expand_min_cosine", default=0.98, type=float)
    parser.add_argument("--full_set_sparse_validation_risk_expand_min_spatial_distance_m", default=1.0, type=float)
    parser.add_argument("--full_set_sparse_validation_risk_expand_weight", default=1.0, type=float)
    parser.add_argument("--full_set_sparse_validation_conflict_max_landmarks_per_query", default=0, type=int)
    parser.add_argument("--full_set_final_prune_no_query_utility", action="store_true")
    parser.add_argument("--full_set_final_prune_conflict_threshold", default=0.0, type=float)
    parser.add_argument("--full_set_final_prune_min_keep_count", default=0, type=int)
    parser.add_argument("--full_set_final_prune_query_match_risk_reference", action="store_true")
    parser.add_argument("--full_set_final_prune_nonreference_query_match_competition", action="store_true")
    parser.add_argument("--full_set_precision_fill_target_count", default=0, type=int)
    parser.add_argument("--full_set_precision_fill_min_score", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_kc_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_visibility_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_mask_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_solver_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_query_support_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_support_match_strength_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_ambiguity_risk_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_sparse_validation_risk_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_match_competition_risk_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_descriptor_conflict_weight", default=0.0, type=float)
    parser.add_argument("--full_set_precision_fill_require_query_support", action="store_true")
    parser.add_argument("--full_set_precision_fill_max_per_spatial_cluster", default=0, type=int)
    parser.add_argument("--full_set_preserve_source_descriptors", action="store_true")
    parser.add_argument("--full_set_validation_min_query_landmarks", default=0, type=int)
    parser.add_argument("--full_set_validation_min_query_cells", default=0, type=int)
    parser.add_argument("--full_set_validation_min_query_depth_bins", default=0, type=int)
    parser.add_argument("--full_set_validation_min_query_bearing_spread", default=0.0, type=float)
    parser.add_argument("--full_set_validation_min_query_camera_spread_m", default=0.0, type=float)
    parser.add_argument("--full_set_local_search_rounds", default=0, type=int)
    parser.add_argument("--full_set_local_search_candidates_per_query", default=0, type=int)
    parser.add_argument("--full_set_local_search_max_additions", default=0, type=int)
    parser.add_argument("--full_set_local_search_reserve_count", default=0, type=int)
    parser.add_argument("--full_set_local_prune_conflict_threshold", default=0.0, type=float)
    parser.add_argument("--full_set_local_prune_validate_queries", action="store_true")
    parser.add_argument("--full_set_augmentation_min_solver_support", default=0.0, type=float)
    parser.add_argument("--full_set_augmentation_min_query_support", default=0.0, type=float)
    parser.add_argument("--full_set_augmentation_min_kc_score", default=0.0, type=float)
    parser.add_argument("--require_candidate_gain", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    ulf_root = Path(args.ulf_root)
    config = yaml.load(Path(args.cfg).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    config = apply_solver_feedback_alpha_overrides(
        config,
        sampling_alpha=args.solver_sampling_alpha,
        fusion_alpha=args.solver_fusion_alpha,
    )
    log_name = str(config.get("log_name", "test"))
    solver_feedback_path, solver_feedback_resolution = resolve_solver_feedback_path_for_resample(
        explicit_feedback_path=args.solver_feedback,
        source_model_path=Path(args.source_model_path),
        log_name=log_name,
        config=config,
    )
    if solver_feedback_path is None and args.native_sampled_idx is not None:
        raise ValueError("--native_sampled_idx requires --solver_feedback")
    if str(args.selection_policy) == "full_gaussian_sparse_set":
        if args.native_sampled_idx is not None:
            raise ValueError("full_gaussian_sparse_set selects from full Gaussians and must not use --native_sampled_idx")
        if (
            solver_feedback_path is None
            and args.ray_solver_feedback is None
            and args.full_set_sparse_validation_profile is None
        ):
            raise ValueError(
                "full_gaussian_sparse_set requires solver evidence: provide --full_set_sparse_validation_profile "
                "(preferred attributed sparse-PnP trace), --ray_solver_feedback, or legacy --solver_feedback"
            )
        if (
            args.solver_admissibility_path is None
            and args.ray_solver_feedback is None
            and args.full_set_sparse_validation_profile is None
        ):
            raise ValueError(
                "full_gaussian_sparse_set requires query-level solver evidence: provide "
                "--full_set_sparse_validation_profile, --ray_solver_feedback, or legacy --solver_admissibility_path"
            )
    paths = prepare_ulfloc_solver_feedback_output(
        source_model_path=Path(args.source_model_path),
        output_model_path=Path(args.output_model_path),
        feedback_path=solver_feedback_path,
        log_name=log_name,
        iteration=int(args.iteration),
        force=bool(args.force),
    )
    if solver_feedback_path is not None:
        validation = validate_solver_feedback_matches_source_map(
            feedback_path=solver_feedback_path,
            source_model_path=Path(args.source_model_path),
            iteration=int(args.iteration),
        )
    else:
        source_count = read_ply_vertex_count(_iteration_dir(Path(args.source_model_path), int(args.iteration)) / "point_cloud.ply")
        validation = {
            "source_gaussian_count": int(source_count),
            "feedback_weight_count": None,
            "feedback_split_name": "none",
            "feedback_schema_version": "none",
        }

    imports = _import_ulfloc_postprocess(ulf_root)
    original_source_path = Path(args.source_path)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        original_source_path,
        Path(args.output_model_path) / "_ulf_loader_links",
    )
    dataset_args = argparse.Namespace(
        sh_degree=int(args.sh_degree),
        source_path=loader_source_path,
        feature_type=str(args.feature_type or config.get("feature_type", "sp")),
        gaussian_type=str(args.gaussian_type),
        model_path=Path(args.source_model_path),
        images=str(args.images),
        resolution=int(args.resolution),
        longest_edge=int(args.longest_edge),
        data_device=str(args.data_device),
    )
    dataset = _dataset_namespace(dataset_args, config)
    train_images_to_read, train_list_path = resolve_train_images_to_read(
        Path(dataset.source_path),
        Path(args.train_list) if args.train_list is not None else None,
    )
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = str(args.output_model_path)
    if args.sample_random_seed is not None:
        config.setdefault("sample", {})["random_seed"] = int(args.sample_random_seed)

    if dataset.gaussian_type == "3dgs":
        gaussians = imports["GaussianModel"](dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = imports["GaussianModel_2dgs"](dataset.sh_degree)
    else:
        raise ValueError(f"unsupported gaussian_type: {dataset.gaussian_type}")
    scene_obj = imports["Scene"](
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        images_to_read=train_images_to_read,
        preload_cameras=True,
    )
    masks = None if args.no_masks else _load_masks(Path(dataset.source_path), dataset.images)
    feature_extractor = imports["FeatureExtractor"](config["feature_type"]).cuda().eval()
    output_eval_dir = paths["output_eval_dir"]
    reused_sampled_idx_summary: dict[str, Any] | None = None
    if args.reuse_sampled_idx is not None:
        if str(args.selection_policy) == "full_gaussian_sparse_set":
            raise ValueError("--reuse_sampled_idx cannot be combined with full_gaussian_sparse_set")
        reused_sampled_idx_summary = prepare_reused_sampled_idx(
            source_sampled_idx=Path(args.reuse_sampled_idx),
            output_eval_dir=output_eval_dir,
            landmark_file_name=str(config["sample"]["landmark_file_name"]),
            force=bool(args.force),
        )
    full_sparse_set_summary: dict[str, Any] | None = None
    if str(args.selection_policy) == "full_gaussian_sparse_set":
        feedback_payload = {} if solver_feedback_path is None else _load_solver_feedback(Path(solver_feedback_path))
        solver_constraints = (
            {}
            if args.solver_admissibility_path is None
            else _load_solver_constraints(Path(args.solver_admissibility_path))
        )
        count = int(gaussians.get_xyz.shape[0])
        ray_feedback_payload = (
            None
            if args.ray_solver_feedback is None
            else _load_ray_solver_feedback(Path(args.ray_solver_feedback), num_gaussians=count)
        )
        sparse_validation_profile = _load_sparse_validation_profile(args.full_set_sparse_validation_profile)
        sparse_validation_selector_metadata: dict[str, Any] = {}
        if sparse_validation_profile is None:
            sparse_validation_risk = torch.zeros(count, dtype=torch.float32)
            sparse_validation_protected_support: dict[str, dict[int, float]] = {}
            sparse_validation_observations: dict[str, dict[int, dict[str, Any]]] = {}
            sparse_validation_hard_reject_exempt = torch.zeros(count, dtype=torch.bool)
        else:
            sparse_validation_selector_signals = sparse_validation_signals_from_pnp_profile(
                sparse_validation_profile,
                num_gaussians=count,
                support_scope=str(args.full_set_sparse_validation_support_scope),
                require_attribution=not bool(args.full_set_allow_metric_only_sparse_validation_profile),
            )
            sparse_validation_risk = sparse_validation_selector_signals["sparse_validation_risk"]
            sparse_validation_protected_support = sparse_validation_selector_signals["validation_per_query_support"]
            sparse_validation_observations = sparse_validation_selector_signals["per_query_observations"]
            sparse_validation_hard_reject_exempt = sparse_validation_selector_signals[
                "sparse_validation_hard_reject_exempt"
            ]
            sparse_validation_selector_metadata = dict(sparse_validation_selector_signals.get("metadata", {}))
        full_signals = compute_full_gaussian_kc_visibility_signals(
            feature_extractor=feature_extractor,
            scene_obj=scene_obj,
            gaussians=gaussians,
            masks=masks,
            config=config,
            get_render_visible_mask=imports["get_render_visible_mask"],
            collect_descriptor_proxy=full_set_needs_descriptor_features(args),
        )
        solver_support = _optional_feedback_vector(
            feedback_payload,
            ("landmark_support", "support_score", "solver_support", "hard_gain"),
            count,
        )
        validation_solver_support = _aggregate_per_query_support_vector(
            sparse_validation_protected_support,
            length=count,
        )
        if float(solver_support.abs().sum().item()) == 0.0 and "landmark_weights" in feedback_payload:
            solver_support = torch.as_tensor(feedback_payload["landmark_weights"], dtype=torch.float32).reshape(-1).cpu()
        if float(validation_solver_support.abs().sum().item()) > 0.0:
            solver_support = torch.maximum(solver_support, validation_solver_support)
        if ray_feedback_payload is not None and "support_score" in ray_feedback_payload:
            ray_support = torch.as_tensor(ray_feedback_payload["support_score"], dtype=torch.float32).reshape(-1).cpu()
            solver_support = torch.maximum(solver_support, ray_support)
        ambiguity_risk = _optional_feedback_vector(
            feedback_payload,
            ("ambiguity_risk", "landmark_ambiguity_risk", "hard_negative_risk", "negative_support_risk", "landmark_risk"),
            count,
        )
        if ray_feedback_payload is not None:
            for risk_key in ("hard_negative_risk", "artifact_risk"):
                if risk_key not in ray_feedback_payload:
                    continue
                ray_risk = torch.as_tensor(ray_feedback_payload[risk_key], dtype=torch.float32).reshape(-1).cpu()
                ambiguity_risk = torch.maximum(ambiguity_risk, ray_risk)
        support_match_strength = _load_optional_full_gaussian_vector(
            args.full_set_support_match_strength,
            num_gaussians=count,
            name="support_match_strength",
        )
        match_competition_risk = _load_optional_full_gaussian_vector(
            args.full_set_match_competition_risk,
            num_gaussians=count,
            name="match_competition_risk",
        )
        per_query_match_strength, per_query_match_competition_risk = _load_query_match_dominance(
            args.full_set_query_match_dominance,
            num_gaussians=count,
        )
        query_match_reference_sampled_idx = (
            None
            if args.full_set_query_match_reference_sampled_idx is None
            else _load_sampled_idx(Path(args.full_set_query_match_reference_sampled_idx))
        )
        per_query_match_competition_risk_reference = _query_match_risk_reference_from_sampled_idx(
            per_query_match_competition_risk,
            query_match_reference_sampled_idx,
        )
        per_query_match_competition_reference_landmarks = (
            _query_match_competition_reference_landmarks_from_sampled_idx(
                per_query_match_competition_risk,
                query_match_reference_sampled_idx,
            )
        )
        source_anchor_idx = (
            None if args.full_set_source_anchor_idx is None else _load_sampled_idx(Path(args.full_set_source_anchor_idx))
        )
        per_query_support = (
            {}
            if not solver_constraints
            else per_query_support_from_solver_constraints(solver_constraints)
        )
        if ray_feedback_payload is not None:
            per_query_support = merge_per_query_support(
                per_query_support,
                per_query_support_from_ray_feedback(ray_feedback_payload),
            )
        per_query_observations = (
            {}
            if ray_feedback_payload is None
            else per_query_observations_from_ray_feedback(ray_feedback_payload)
        )
        per_query_observations = merge_per_query_observations(
            sparse_validation_observations,
            per_query_observations,
        )
        explicit_descriptor_conflict_edges = _load_descriptor_conflict_edges(args.full_set_descriptor_conflict_graph)
        sparse_validation_conflict_edges = (
            {}
            if sparse_validation_profile is None
            or int(args.full_set_sparse_validation_conflict_max_landmarks_per_query) <= 0
            else descriptor_conflict_edges_from_sparse_validation_profile(
                sparse_validation_profile,
                max_landmarks_per_query=int(args.full_set_sparse_validation_conflict_max_landmarks_per_query),
            )
        )
        descriptor_conflict_edges = _merge_conflict_edge_tables(
            explicit_descriptor_conflict_edges,
            sparse_validation_conflict_edges,
        )
        descriptor_features = None
        descriptor_feature_status = "disabled"
        if full_set_needs_descriptor_features(args):
            descriptor_feature_status = "missing"
            raw_descriptor_features = getattr(gaussians, "get_loc_feature", None)
            if raw_descriptor_features is None:
                raw_descriptor_features = getattr(gaussians, "_loc_feature", None)
            if callable(raw_descriptor_features):
                raw_descriptor_features = raw_descriptor_features()
            if raw_descriptor_features is not None:
                candidate_features = torch.as_tensor(raw_descriptor_features).detach()
                if candidate_features.numel() > 0 and int(candidate_features.shape[0]) == int(count):
                    descriptor_features = candidate_features.flatten(start_dim=1).cpu()
                    descriptor_feature_status = "native_loc_feature"
            if descriptor_features is None and "descriptor_proxy" in full_signals:
                descriptor_features = torch.as_tensor(full_signals["descriptor_proxy"], dtype=torch.float32).cpu()
                descriptor_feature_status = "train_keypoint_proxy"
            if descriptor_features is None:
                print(
                    "Full-set descriptor features were requested but full-Gaussian loc descriptors are unavailable; "
                    "continuing without descriptor-derived conflict/risk expansion."
                )
        sparse_result = select_sparse_solver_set_from_full_gaussians(
            SparseSetSignals(
                xyz=gaussians.get_xyz.detach().cpu(),
                kc_score=full_signals["kc_score"],
                visibility_score=full_signals["visibility_score"],
                mask_validity=full_signals["mask_validity"],
                solver_support=solver_support,
                per_query_support=per_query_support,
                validation_per_query_support=sparse_validation_protected_support,
                per_query_match_strength=per_query_match_strength,
                per_query_match_competition_risk=per_query_match_competition_risk,
                per_query_match_competition_risk_reference=per_query_match_competition_risk_reference,
                per_query_match_competition_reference_landmarks=per_query_match_competition_reference_landmarks,
                per_query_observations=per_query_observations,
                descriptor_conflict_edges=descriptor_conflict_edges,
                descriptor_features=descriptor_features,
                ambiguity_risk=ambiguity_risk,
                sparse_validation_risk=sparse_validation_risk,
                sparse_validation_hard_reject_exempt=sparse_validation_hard_reject_exempt,
                support_match_strength=support_match_strength,
                match_competition_risk=match_competition_risk,
                source_anchor_idx=source_anchor_idx,
            ),
            SparseSetSelectionConfig(
                max_landmarks=int(args.full_set_max_landmarks),
                min_landmarks=int(args.full_set_min_landmarks),
                min_marginal_gain=float(args.full_set_min_marginal_gain),
                candidate_pool_size=int(args.full_set_candidate_pool_size),
                candidate_pool_kc_top_k=int(args.full_set_candidate_pool_kc_top_k),
                candidate_pool_visibility_top_k=int(args.full_set_candidate_pool_visibility_top_k),
                candidate_pool_mask_top_k=int(args.full_set_candidate_pool_mask_top_k),
                candidate_pool_solver_top_k=int(args.full_set_candidate_pool_solver_top_k),
                candidate_pool_match_strength_top_k=int(args.full_set_candidate_pool_match_strength_top_k),
                candidate_pool_query_match_strength_top_k=int(
                    args.full_set_candidate_pool_query_match_strength_top_k
                ),
                candidate_pool_query_top_k=int(args.full_set_candidate_pool_query_top_k),
                candidate_pool_query_cell_top_k=int(args.full_set_candidate_pool_query_cell_top_k),
                candidate_pool_query_depth_top_k=int(args.full_set_candidate_pool_query_depth_top_k),
                target_query_support=float(args.full_set_target_query_support),
                spatial_cluster_size_m=float(args.coverage_grid_size_m),
                max_per_spatial_cluster=int(args.full_set_max_per_spatial_cluster),
                min_visibility=float(args.full_set_min_visibility),
                min_mask_validity=float(args.full_set_min_mask_validity),
                kc_weight=float(args.kc_rank_weight),
                visibility_weight=float(args.full_set_visibility_weight),
                mask_weight=float(args.full_set_mask_weight),
                solver_weight=float(args.full_set_solver_weight),
                query_support_weight=float(args.full_set_query_support_weight),
                geometry_weight=float(args.full_set_geometry_weight),
                depth_weight=float(args.full_set_depth_weight),
                query_geometry_weight=float(args.full_set_query_geometry_weight),
                query_depth_weight=float(args.full_set_query_depth_weight),
                query_bearing_weight=float(args.full_set_query_bearing_weight),
                query_pnp_geometry_weight=float(args.full_set_query_pnp_geometry_weight),
                query_pnp_distance_scale_m=float(args.full_set_query_pnp_distance_scale_m),
                ambiguity_risk_weight=float(args.ambiguity_risk_weight),
                sparse_validation_risk_weight=float(args.full_set_sparse_validation_risk_weight),
                sparse_validation_risk_reject_threshold=float(
                    args.full_set_sparse_validation_risk_reject_threshold
                ),
                sparse_validation_risk_protected_hard_reject_exempt=bool(
                    args.full_set_sparse_validation_protected_hard_reject_exempt
                ),
                sparse_validation_risk_source_anchor_exempt=bool(
                    args.full_set_sparse_validation_source_anchor_risk_exempt
                ),
                support_match_strength_weight=float(args.full_set_support_match_strength_weight),
                match_competition_risk_weight=float(args.full_set_match_competition_risk_weight),
                match_competition_risk_reject_threshold=float(
                    args.full_set_match_competition_risk_reject_threshold
                ),
                query_match_strength_weight=float(args.full_set_query_match_strength_weight),
                query_match_competition_weight=float(args.full_set_query_match_competition_weight),
                query_match_competition_ratio_weight=float(
                    args.full_set_query_match_competition_ratio_weight
                ),
                query_match_risk_reference_margin=float(args.full_set_query_match_risk_reference_margin),
                query_match_risk_reference_weight=float(args.full_set_query_match_risk_reference_weight),
                query_match_nonreference_competition_weight=float(
                    args.full_set_query_match_nonreference_competition_weight
                ),
                sparse_validation_risk_expand_top_k=int(args.full_set_sparse_validation_risk_expand_top_k),
                sparse_validation_risk_expand_neighbors_per_seed=int(
                    args.full_set_sparse_validation_risk_expand_neighbors_per_seed
                ),
                sparse_validation_risk_expand_candidate_limit=int(
                    args.full_set_sparse_validation_risk_expand_candidate_limit
                ),
                sparse_validation_risk_expand_min_cosine=float(
                    args.full_set_sparse_validation_risk_expand_min_cosine
                ),
                sparse_validation_risk_expand_min_spatial_distance_m=float(
                    args.full_set_sparse_validation_risk_expand_min_spatial_distance_m
                ),
                sparse_validation_risk_expand_weight=float(args.full_set_sparse_validation_risk_expand_weight),
                easy_query_gain_decay=float(args.easy_query_gain_decay),
                query_prefill_target_fraction=float(args.full_set_query_prefill_target_fraction),
                query_prefill_max_candidates_per_query=int(args.full_set_query_prefill_max_candidates_per_query),
                query_prefill_max_per_spatial_cluster=int(args.full_set_query_prefill_max_per_spatial_cluster),
                query_prefill_min_cells_per_query=int(args.full_set_query_prefill_min_cells_per_query),
                query_prefill_min_depth_bins_per_query=int(args.full_set_query_prefill_min_depth_bins_per_query),
                query_prefill_hard_coverage_bonus=float(args.full_set_query_prefill_hard_coverage_bonus),
                source_query_prefill_target_fraction=float(args.full_set_source_query_prefill_target_fraction),
                source_query_prefill_max_candidates_per_query=int(
                    args.full_set_source_query_prefill_max_candidates_per_query
                ),
                source_query_prefill_max_per_spatial_cluster=int(
                    args.full_set_source_query_prefill_max_per_spatial_cluster
                ),
                source_query_prefill_min_cells_per_query=int(
                    args.full_set_source_query_prefill_min_cells_per_query
                ),
                source_query_prefill_min_depth_bins_per_query=int(
                    args.full_set_source_query_prefill_min_depth_bins_per_query
                ),
                validation_query_prefill_target_fraction=float(
                    args.full_set_validation_query_prefill_target_fraction
                ),
                validation_query_prefill_max_candidates_per_query=int(
                    args.full_set_validation_query_prefill_max_candidates_per_query
                ),
                validation_query_prefill_max_per_spatial_cluster=int(
                    args.full_set_validation_query_prefill_max_per_spatial_cluster
                ),
                validation_query_prefill_min_cells_per_query=int(
                    args.full_set_validation_query_prefill_min_cells_per_query
                ),
                validation_query_prefill_min_depth_bins_per_query=int(
                    args.full_set_validation_query_prefill_min_depth_bins_per_query
                ),
                validation_query_prefill_force_all=bool(args.full_set_validation_query_prefill_force_all),
                main_min_query_gain=float(args.full_set_main_min_query_gain),
                main_dynamic_lookahead=int(args.full_set_main_dynamic_lookahead),
                kc_anchor_count=int(args.full_set_kc_anchor_count),
                kc_anchor_min_score=float(args.full_set_kc_anchor_min_score),
                kc_anchor_max_per_spatial_cluster=int(args.full_set_kc_anchor_max_per_spatial_cluster),
                source_anchor_mode=str(args.full_set_source_anchor_mode),
                source_anchor_weight=float(args.full_set_source_anchor_weight),
                force_source_anchor=bool(args.full_set_force_source_anchor),
                descriptor_conflict_weight=float(args.full_set_descriptor_conflict_weight),
                auto_descriptor_conflict_top_k=int(args.full_set_auto_descriptor_conflict_top_k),
                auto_descriptor_conflict_candidate_limit=int(args.full_set_auto_descriptor_conflict_candidate_limit),
                auto_descriptor_conflict_min_cosine=float(args.full_set_auto_descriptor_conflict_min_cosine),
                auto_descriptor_conflict_min_spatial_distance_m=float(
                    args.full_set_auto_descriptor_conflict_min_spatial_distance_m
                ),
                final_prune_no_query_utility=bool(args.full_set_final_prune_no_query_utility),
                final_prune_conflict_threshold=float(args.full_set_final_prune_conflict_threshold),
                final_prune_min_keep_count=int(args.full_set_final_prune_min_keep_count),
                final_prune_query_match_risk_reference=bool(
                    args.full_set_final_prune_query_match_risk_reference
                ),
                final_prune_nonreference_query_match_competition=bool(
                    args.full_set_final_prune_nonreference_query_match_competition
                ),
                precision_fill_target_count=int(args.full_set_precision_fill_target_count),
                precision_fill_min_score=float(args.full_set_precision_fill_min_score),
                precision_fill_kc_weight=float(args.full_set_precision_fill_kc_weight),
                precision_fill_visibility_weight=float(args.full_set_precision_fill_visibility_weight),
                precision_fill_mask_weight=float(args.full_set_precision_fill_mask_weight),
                precision_fill_solver_weight=float(args.full_set_precision_fill_solver_weight),
                precision_fill_query_support_weight=float(
                    args.full_set_precision_fill_query_support_weight
                ),
                precision_fill_support_match_strength_weight=float(
                    args.full_set_precision_fill_support_match_strength_weight
                ),
                precision_fill_ambiguity_risk_weight=float(
                    args.full_set_precision_fill_ambiguity_risk_weight
                ),
                precision_fill_sparse_validation_risk_weight=float(
                    args.full_set_precision_fill_sparse_validation_risk_weight
                ),
                precision_fill_match_competition_risk_weight=float(
                    args.full_set_precision_fill_match_competition_risk_weight
                ),
                precision_fill_descriptor_conflict_weight=float(
                    args.full_set_precision_fill_descriptor_conflict_weight
                ),
                precision_fill_require_query_support=bool(
                    args.full_set_precision_fill_require_query_support
                ),
                precision_fill_max_per_spatial_cluster=int(
                    args.full_set_precision_fill_max_per_spatial_cluster
                ),
                validation_min_query_landmarks=int(args.full_set_validation_min_query_landmarks),
                validation_min_query_cells=int(args.full_set_validation_min_query_cells),
                validation_min_query_depth_bins=int(args.full_set_validation_min_query_depth_bins),
                validation_min_query_bearing_spread=float(args.full_set_validation_min_query_bearing_spread),
                validation_min_query_camera_spread_m=float(args.full_set_validation_min_query_camera_spread_m),
                local_search_rounds=int(args.full_set_local_search_rounds),
                local_search_candidates_per_query=int(args.full_set_local_search_candidates_per_query),
                local_search_max_additions=int(args.full_set_local_search_max_additions),
                local_search_reserve_count=int(args.full_set_local_search_reserve_count),
                local_prune_conflict_threshold=float(args.full_set_local_prune_conflict_threshold),
                local_prune_validate_queries=bool(args.full_set_local_prune_validate_queries),
                augmentation_min_solver_support=float(args.full_set_augmentation_min_solver_support),
                augmentation_min_query_support=float(args.full_set_augmentation_min_query_support),
                augmentation_min_kc_score=float(args.full_set_augmentation_min_kc_score),
            ),
        )
        sampled_idx = sparse_result.sampled_idx
        if int(sampled_idx.numel()) <= 0:
            raise ValueError("full_gaussian_sparse_set selected zero landmarks; lower min thresholds or inspect signals")
        sampled_path = output_eval_dir / config["sample"]["landmark_file_name"]
        with sampled_path.open("wb") as handle:
            pickle.dump(sampled_idx, handle)
        full_sparse_set_summary = {
            **sparse_result.metadata,
            "solver_admissibility_path": _optional_path_string(args.solver_admissibility_path),
            "ray_solver_feedback_path": _optional_path_string(args.ray_solver_feedback),
            "sparse_validation_profile_path": _optional_path_string(args.full_set_sparse_validation_profile),
            "sparse_validation_profile_schema": None
            if sparse_validation_profile is None
            else str(sparse_validation_profile.get("schema", "")),
            "sparse_validation_attribution_status": None
            if sparse_validation_profile is None
            else str(sparse_validation_profile.get("attribution_status", "unknown")),
            "sparse_validation_profile_split_name": None
            if sparse_validation_profile is None
            else str(sparse_validation_profile.get("split_name", "")),
            "sparse_validation_allow_metric_only": bool(
                args.full_set_allow_metric_only_sparse_validation_profile
            ),
            "sparse_validation_support_scope": str(args.full_set_sparse_validation_support_scope),
            "sparse_validation_selector_metadata": sparse_validation_selector_metadata,
            "sparse_validation_landmark_risk_count": int((sparse_validation_risk > 0).sum().item()),
            "sparse_validation_landmark_risk_max": float(sparse_validation_risk.max().item())
            if int(sparse_validation_risk.numel()) > 0
            else 0.0,
            "sparse_validation_solver_support_nonzero_count": int(
                (validation_solver_support > 0.0).sum().item()
            ),
            "sparse_validation_solver_support_max": float(validation_solver_support.max().item())
            if int(validation_solver_support.numel()) > 0
            else 0.0,
            "sparse_validation_protected_query_count": int(len(sparse_validation_protected_support)),
            "sparse_validation_protected_support_entry_count": int(
                sum(len(support) for support in sparse_validation_protected_support.values())
            ),
            "sparse_validation_conflict_max_landmarks_per_query": int(
                args.full_set_sparse_validation_conflict_max_landmarks_per_query
            ),
            "sparse_validation_conflict_edge_count": int(
                sum(len(neighbors) for neighbors in sparse_validation_conflict_edges.values())
            ),
            "support_match_strength_path": None
            if args.full_set_support_match_strength is None
            else str(Path(args.full_set_support_match_strength)),
            "support_match_strength_nonzero_count": int((support_match_strength > 0.0).sum().item()),
            "support_match_strength_max": float(support_match_strength.max().item())
            if int(support_match_strength.numel()) > 0
            else 0.0,
            "match_competition_risk_path": None
            if args.full_set_match_competition_risk is None
            else str(Path(args.full_set_match_competition_risk)),
            "match_competition_risk_nonzero_count": int((match_competition_risk > 0.0).sum().item()),
            "match_competition_risk_max": float(match_competition_risk.max().item())
            if int(match_competition_risk.numel()) > 0
            else 0.0,
            "query_match_dominance_path": None
            if args.full_set_query_match_dominance is None
            else str(Path(args.full_set_query_match_dominance)),
            "query_match_strength_weight": float(args.full_set_query_match_strength_weight),
            "query_match_competition_weight": float(args.full_set_query_match_competition_weight),
            "query_match_competition_ratio_weight": float(args.full_set_query_match_competition_ratio_weight),
            "query_match_reference_sampled_idx": None
            if args.full_set_query_match_reference_sampled_idx is None
            else str(Path(args.full_set_query_match_reference_sampled_idx)),
            "query_match_risk_reference_margin": float(args.full_set_query_match_risk_reference_margin),
            "query_match_risk_reference_weight": float(args.full_set_query_match_risk_reference_weight),
            "query_match_risk_reference_query_count": int(len(per_query_match_competition_risk_reference)),
            "query_match_competition_reference_query_count": int(
                len(per_query_match_competition_reference_landmarks)
            ),
            "query_match_competition_reference_entry_count": int(
                sum(len(values) for values in per_query_match_competition_reference_landmarks.values())
            ),
            "query_match_nonreference_competition_weight": float(
                args.full_set_query_match_nonreference_competition_weight
            ),
            "query_match_strength_query_count": int(len(per_query_match_strength)),
            "query_match_strength_entry_count": int(sum(len(scores) for scores in per_query_match_strength.values())),
            "query_match_competition_risk_query_count": int(len(per_query_match_competition_risk)),
            "query_match_competition_risk_entry_count": int(
                sum(len(scores) for scores in per_query_match_competition_risk.values())
            ),
            "ray_per_query_support_query_count": int(
                len(per_query_support_from_ray_feedback(ray_feedback_payload)) if ray_feedback_payload is not None else 0
            ),
            "ray_per_query_observation_query_count": int(len(per_query_observations)),
            "ray_per_query_observation_entry_count": int(sum(len(values) for values in per_query_observations.values())),
            "sparse_validation_observation_query_count": int(len(sparse_validation_observations)),
            "sparse_validation_observation_entry_count": int(
                sum(len(values) for values in sparse_validation_observations.values())
            ),
            "merged_per_query_support_query_count": int(len(per_query_support)),
            "merged_per_query_support_entry_count": int(sum(len(support) for support in per_query_support.values())),
            "descriptor_features_used_for_auto_conflict": descriptor_features is not None,
            "descriptor_feature_status_for_auto_conflict": descriptor_feature_status,
            "kc_nonzero_count": int((full_signals["kc_score"] > 0).sum().item()),
            "visibility_nonzero_count": int((full_signals["visibility_score"] > 0).sum().item()),
            "mask_valid_mean_observed": float(
                full_signals["mask_validity"][full_signals["visibility_score"] > 0].mean().item()
            )
            if bool((full_signals["visibility_score"] > 0).any())
            else 0.0,
        }
        print(
            "Applied full-Gaussian sparse set selection: "
            f"selected={full_sparse_set_summary['selected_count']} / {full_sparse_set_summary['input_gaussian_count']}, "
            f"candidate_pool={full_sparse_set_summary['candidate_count']}"
        )
    else:
        sampled_idx = imports["keypoints_vote"](
            feature_extractor,
            scene_obj,
            gaussians,
            masks,
            str(output_eval_dir),
            config,
        )
    native_safe_core_enabled = args.native_sampled_idx is not None
    native_safe_core_summary: dict[str, Any] | None = None
    if native_safe_core_enabled:
        native_sampled_idx = _load_sampled_idx(Path(args.native_sampled_idx))
        feedback_payload = _load_solver_feedback(solver_feedback_path)
        solver_weights = torch.as_tensor(feedback_payload["landmark_weights"], dtype=torch.float32).reshape(-1).cpu()
        original_sampled_idx = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
        ranking_scores = solver_weights
        selection_policy_summary: dict[str, Any] = {"selection_policy": "solver_weight"}
        if str(args.selection_policy) in {"kc_solver", "solver_coverage", "solver_query_coverage", "sparse_solver_aware_kc"}:
            score_component_summary = summarize_solver_feedback_score_components(
                feedback_payload,
                support_weight=float(args.support_weight),
                regression_risk_weight=float(args.regression_risk_weight),
                dense_worsen_risk_weight=float(args.dense_worsen_risk_weight),
                ambiguity_risk_weight=float(args.ambiguity_risk_weight),
                artifact_risk_weight=float(args.artifact_risk_weight),
                hard_negative_risk_weight=float(args.hard_negative_risk_weight),
            )
            ranking_scores = build_kc_solver_selection_scores(
                feedback_payload,
                candidate_sampled_idx=original_sampled_idx,
                solver_gain_weight=float(args.solver_gain_weight),
                kc_rank_weight=float(args.kc_rank_weight),
                support_weight=float(args.support_weight),
                regression_risk_weight=float(args.regression_risk_weight),
                dense_worsen_risk_weight=float(args.dense_worsen_risk_weight),
                ambiguity_risk_weight=float(args.ambiguity_risk_weight),
                artifact_risk_weight=float(args.artifact_risk_weight),
                hard_negative_risk_weight=float(args.hard_negative_risk_weight),
            )
            selection_policy_summary = {
                "selection_policy": str(args.selection_policy),
                "kc_rank_weight": float(args.kc_rank_weight),
                "solver_gain_weight": float(args.solver_gain_weight),
                "support_weight": float(args.support_weight),
                "regression_risk_weight": float(args.regression_risk_weight),
                "dense_worsen_risk_weight": float(args.dense_worsen_risk_weight),
                "ambiguity_risk_weight": float(args.ambiguity_risk_weight),
                "artifact_risk_weight": float(args.artifact_risk_weight),
                "hard_negative_risk_weight": float(args.hard_negative_risk_weight),
                "score_components": score_component_summary,
            }
        if str(args.selection_policy) == "sparse_solver_aware_kc":
            if args.solver_admissibility_path is None:
                raise ValueError("--solver_admissibility_path is required for selection_policy=sparse_solver_aware_kc")
            solver_constraints = _load_solver_constraints(Path(args.solver_admissibility_path))
            cluster_ids = build_spatial_cluster_ids(
                gaussians.get_xyz.detach().cpu(),
                grid_size_m=float(args.coverage_grid_size_m),
            )
            sparse_result = select_sparse_solver_aware_kc_sampled_idx(
                native_sampled_idx,
                original_sampled_idx,
                solver_scores=ranking_scores,
                solver_constraints=solver_constraints,
                target_count=int(original_sampled_idx.numel()),
                cluster_ids=cluster_ids,
                max_per_cluster=int(args.max_insertions_per_cluster),
                solver_score_weight=float(args.sparse_solver_score_weight),
                hard_query_weight=float(args.sparse_hard_query_weight),
                source_protection_weight=float(args.sparse_source_protection_weight),
                coverage_saturation_mode=str(args.coverage_saturation_mode),
                coverage_saturation_percentile=float(args.coverage_saturation_percentile),
                coverage_saturation_fraction=float(args.coverage_saturation_fraction),
                easy_query_gain_decay=float(args.easy_query_gain_decay),
            )
            merged_sampled_idx = sparse_result.sampled_idx
            selection_policy_summary.update(
                {
                    "solver_admissibility_path": str(Path(args.solver_admissibility_path)),
                    "sparse_solver_aware_kc": sparse_result.metadata,
                    "coverage_grid_size_m": float(args.coverage_grid_size_m),
                    "max_per_cluster": int(args.max_insertions_per_cluster),
                    "sparse_solver_score_weight": float(args.sparse_solver_score_weight),
                    "sparse_hard_query_weight": float(args.sparse_hard_query_weight),
                    "sparse_source_protection_weight": float(args.sparse_source_protection_weight),
                    "coverage_saturation_mode": str(args.coverage_saturation_mode),
                    "coverage_saturation_percentile": float(args.coverage_saturation_percentile),
                    "coverage_saturation_fraction": float(args.coverage_saturation_fraction),
                    "easy_query_gain_decay": float(args.easy_query_gain_decay),
                }
            )
        elif str(args.selection_policy) == "solver_query_coverage":
            if args.solver_admissibility_path is None:
                raise ValueError("--solver_admissibility_path is required for selection_policy=solver_query_coverage")
            solver_constraints = _load_solver_constraints(Path(args.solver_admissibility_path))
            query_coverage_result = merge_solver_query_coverage_sampled_idx(
                native_sampled_idx,
                original_sampled_idx,
                solver_scores=ranking_scores,
                solver_constraints=solver_constraints,
                safe_core_fraction=float(args.native_safe_core_fraction),
                max_solver_insertions=args.max_solver_insertions,
                max_drop_scan=int(args.query_coverage_max_drop_scan),
                min_coverage_gain=float(args.query_coverage_min_coverage_gain),
                min_query_coverage=float(args.query_coverage_min_query_coverage),
                coverage_saturation_mode=str(args.coverage_saturation_mode),
                coverage_saturation_percentile=float(args.coverage_saturation_percentile),
                coverage_saturation_fraction=float(args.coverage_saturation_fraction),
                easy_query_gain_decay=float(args.easy_query_gain_decay),
                tail_cvar_alpha=float(args.tail_cvar_alpha),
                tail_query_gain_boost=float(args.tail_query_gain_boost),
                hard_query_min_gain=float(args.hard_query_min_gain),
                saturation_drop_protection_weight=float(args.saturation_drop_protection_weight),
                require_candidate_gain=bool(args.require_candidate_gain),
            )
            merged_sampled_idx = query_coverage_result.sampled_idx
            selection_policy_summary.update(
                {
                    "solver_admissibility_path": str(Path(args.solver_admissibility_path)),
                    "query_coverage": query_coverage_result.metadata,
                    "query_coverage_max_drop_scan": int(args.query_coverage_max_drop_scan),
                    "query_coverage_min_coverage_gain": float(args.query_coverage_min_coverage_gain),
                    "query_coverage_min_query_coverage": float(args.query_coverage_min_query_coverage),
                    "coverage_saturation_mode": str(args.coverage_saturation_mode),
                    "coverage_saturation_percentile": float(args.coverage_saturation_percentile),
                    "coverage_saturation_fraction": float(args.coverage_saturation_fraction),
                    "easy_query_gain_decay": float(args.easy_query_gain_decay),
                    "tail_cvar_alpha": float(args.tail_cvar_alpha),
                    "tail_query_gain_boost": float(args.tail_query_gain_boost),
                    "hard_query_min_gain": float(args.hard_query_min_gain),
                    "saturation_drop_protection_weight": float(args.saturation_drop_protection_weight),
                    "require_candidate_gain": bool(args.require_candidate_gain),
                }
            )
        elif str(args.selection_policy) == "solver_coverage":
            cluster_ids = build_spatial_cluster_ids(
                gaussians.get_xyz.detach().cpu(),
                grid_size_m=float(args.coverage_grid_size_m),
            )
            merged_sampled_idx = merge_solver_coverage_coreset_sampled_idx(
                native_sampled_idx,
                original_sampled_idx,
                solver_scores=ranking_scores,
                cluster_ids=cluster_ids,
                safe_core_fraction=float(args.native_safe_core_fraction),
                max_solver_insertions=args.max_solver_insertions,
                max_insertions_per_cluster=int(args.max_insertions_per_cluster),
                min_solver_insertion_score=args.min_solver_insertion_score,
                min_solver_insertion_margin=float(args.min_solver_insertion_margin),
            )
            selection_policy_summary.update(
                {
                    "coverage_grid_size_m": float(args.coverage_grid_size_m),
                    "max_insertions_per_cluster": int(args.max_insertions_per_cluster),
                }
            )
        else:
            merged_sampled_idx = merge_native_safe_core_sampled_idx(
                native_sampled_idx,
                original_sampled_idx,
                solver_weights=ranking_scores,
                safe_core_fraction=float(args.native_safe_core_fraction),
                max_solver_insertions=args.max_solver_insertions,
                min_solver_insertion_score=args.min_solver_insertion_score,
                min_solver_insertion_margin=float(args.min_solver_insertion_margin),
            )
        native_set = set(int(x) for x in native_sampled_idx.tolist())
        original_set = set(int(x) for x in original_sampled_idx.tolist())
        merged_set = set(int(x) for x in merged_sampled_idx.tolist())
        sampled_idx = merged_sampled_idx
        sampled_path = output_eval_dir / config["sample"]["landmark_file_name"]
        with sampled_path.open("wb") as handle:
            pickle.dump(sampled_idx, handle)
        native_safe_core_summary = {
            "native_sampled_idx": str(Path(args.native_sampled_idx)),
            "native_safe_core_fraction": float(args.native_safe_core_fraction),
            "max_solver_insertions": args.max_solver_insertions,
            "min_solver_insertion_score": args.min_solver_insertion_score,
            "min_solver_insertion_margin": float(args.min_solver_insertion_margin),
            "native_overlap_before": int(len(native_set & original_set)),
            "native_overlap_after": int(len(native_set & merged_set)),
            "solver_inserted_count": int(len(merged_set - native_set)),
            "sampled_count": int(sampled_idx.numel()),
            **selection_policy_summary,
        }
        print(
            "Applied native-safe sampled_idx merge: "
            f"native_overlap {native_safe_core_summary['native_overlap_before']} -> "
            f"{native_safe_core_summary['native_overlap_after']}, "
            f"solver_inserted={native_safe_core_summary['solver_inserted_count']}"
        )
    imports["feature_fusion"](
        feature_extractor,
        scene_obj,
        gaussians,
        sampled_idx,
        masks,
        str(output_eval_dir),
        config,
    )

    sampled_path = output_eval_dir / config["sample"]["landmark_file_name"]
    feature_path = output_eval_dir / config["sample"]["kpts_feature_file_name"]
    source_descriptor_preserved_count = 0
    source_descriptor_preservation_enabled = bool(args.full_set_preserve_source_descriptors)
    source_descriptor_preservation_status = "disabled"
    if source_descriptor_preservation_enabled:
        source_eval_dir = Path(args.source_model_path) / str(config["log_name"])
        source_sampled_path = source_eval_dir / config["sample"]["landmark_file_name"]
        source_feature_path = source_eval_dir / config["sample"]["kpts_feature_file_name"]
        if not source_sampled_path.is_file() or not source_feature_path.is_file():
            raise FileNotFoundError(
                "source descriptor preservation requested but source sampled/features are missing: "
                f"{source_sampled_path}, {source_feature_path}"
            )
        current_sampled = _load_sampled_idx(sampled_path)
        with feature_path.open("rb") as handle:
            current_features = pickle.load(handle)
        source_sampled = _load_sampled_idx(source_sampled_path)
        with source_feature_path.open("rb") as handle:
            source_features = pickle.load(handle)
        updated_features, source_descriptor_preserved_count = _preserve_source_descriptors_for_shared_landmarks(
            current_sampled,
            torch.as_tensor(current_features),
            source_sampled,
            torch.as_tensor(source_features),
        )
        with feature_path.open("wb") as handle:
            pickle.dump(updated_features, handle)
        source_descriptor_preservation_status = "applied"
    metrics = {
        **validation,
        "sampled_count": _sampled_count(sampled_path),
        "sampled_path": str(sampled_path),
        "feature_path": str(feature_path),
        "selection_policy": str(args.selection_policy),
        "solver_feedback_enabled": solver_feedback_path is not None,
        "fixed_native_geometry": True,
        "source_descriptor_preservation_enabled": bool(source_descriptor_preservation_enabled),
        "source_descriptor_preservation_status": str(source_descriptor_preservation_status),
        "source_descriptor_preserved_count": int(source_descriptor_preserved_count),
    }
    used_config_path = write_ulfloc_used_config(
        output_eval_dir=output_eval_dir,
        cfg_path=Path(args.cfg),
        config=config,
    )
    metrics["config_path"] = str(used_config_path)
    split_audit = {
        "audit_status": "passed",
        "split_name": validation["feedback_split_name"],
        "test_split_used": False,
        "official_test_used": False,
        "notes": (
            "Solver feedback was produced from train/self-map traces and applied only to "
            "ULF-Loc landmark sampling/feature fusion on the fixed native source map."
            if solver_feedback_path is not None
            else "Native train/self-map postprocess without solver feedback; no official test data was used."
        ),
    }
    manifest = {
        "method": (
            "ulfloc_solver_feedback_fixed_map_resampling"
            if solver_feedback_path is not None
            else "ulfloc_native_fixed_map_postprocess"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "ulf_git_commit": _git_commit(ulf_root),
        "command": " ".join(sys.argv),
        "scene": str(args.scene),
        "split_name": validation["feedback_split_name"],
        "source_path": str(loader_source_path),
        "original_source_path": str(original_source_path),
        "train_list": train_list_path,
        "images_to_read_count": len(train_images_to_read) if train_images_to_read is not None else None,
        "source_model_path": str(args.source_model_path),
        "output_model_path": str(args.output_model_path),
        "solver_feedback": str(solver_feedback_path) if solver_feedback_path is not None else None,
        "solver_feedback_resolution": str(solver_feedback_resolution),
        "ray_solver_feedback": None if args.ray_solver_feedback is None else str(Path(args.ray_solver_feedback)),
        "selection_policy": str(args.selection_policy),
        "cfg": str(args.cfg),
        "exported_cfg": str(used_config_path),
        "iteration": int(args.iteration),
        "hyperparameters": {
            "images": str(args.images),
            "feature_type": str(dataset.feature_type),
            "data_device": str(dataset.data_device),
            "longest_edge": int(dataset.longest_edge),
            "sample_random_seed": int(args.sample_random_seed) if args.sample_random_seed is not None else None,
            "sampling_alpha": float(config.get("solver_feedback", {}).get("sampling_alpha", 0.0)),
            "fusion_alpha": float(config.get("solver_feedback", {}).get("fusion_alpha", 0.0)),
            "native_safe_core_enabled": bool(native_safe_core_enabled),
            "native_safe_core_fraction": float(args.native_safe_core_fraction),
            "max_solver_insertions": args.max_solver_insertions,
            "selection_policy": str(args.selection_policy),
            "ray_solver_feedback": None if args.ray_solver_feedback is None else str(Path(args.ray_solver_feedback)),
            "kc_rank_weight": float(args.kc_rank_weight),
            "solver_gain_weight": float(args.solver_gain_weight),
            "support_weight": float(args.support_weight),
            "regression_risk_weight": float(args.regression_risk_weight),
            "dense_worsen_risk_weight": float(args.dense_worsen_risk_weight),
            "ambiguity_risk_weight": float(args.ambiguity_risk_weight),
            "artifact_risk_weight": float(args.artifact_risk_weight),
            "hard_negative_risk_weight": float(args.hard_negative_risk_weight),
            "coverage_grid_size_m": float(args.coverage_grid_size_m),
            "max_insertions_per_cluster": int(args.max_insertions_per_cluster),
            "solver_admissibility_path": str(Path(args.solver_admissibility_path))
            if args.solver_admissibility_path is not None
            else None,
            "query_coverage_max_drop_scan": int(args.query_coverage_max_drop_scan),
            "query_coverage_min_coverage_gain": float(args.query_coverage_min_coverage_gain),
            "query_coverage_min_query_coverage": float(args.query_coverage_min_query_coverage),
            "coverage_saturation_mode": str(args.coverage_saturation_mode),
            "coverage_saturation_percentile": float(args.coverage_saturation_percentile),
            "coverage_saturation_fraction": float(args.coverage_saturation_fraction),
            "easy_query_gain_decay": float(args.easy_query_gain_decay),
            "tail_cvar_alpha": float(args.tail_cvar_alpha),
            "tail_query_gain_boost": float(args.tail_query_gain_boost),
            "hard_query_min_gain": float(args.hard_query_min_gain),
            "saturation_drop_protection_weight": float(args.saturation_drop_protection_weight),
            "sparse_solver_score_weight": float(args.sparse_solver_score_weight),
            "sparse_hard_query_weight": float(args.sparse_hard_query_weight),
            "sparse_source_protection_weight": float(args.sparse_source_protection_weight),
            "full_set_max_landmarks": int(args.full_set_max_landmarks),
            "full_set_min_landmarks": int(args.full_set_min_landmarks),
            "full_set_min_marginal_gain": float(args.full_set_min_marginal_gain),
            "full_set_candidate_pool_size": int(args.full_set_candidate_pool_size),
            "full_set_candidate_pool_kc_top_k": int(args.full_set_candidate_pool_kc_top_k),
            "full_set_candidate_pool_visibility_top_k": int(args.full_set_candidate_pool_visibility_top_k),
            "full_set_candidate_pool_mask_top_k": int(args.full_set_candidate_pool_mask_top_k),
            "full_set_candidate_pool_solver_top_k": int(args.full_set_candidate_pool_solver_top_k),
            "full_set_candidate_pool_match_strength_top_k": int(
                args.full_set_candidate_pool_match_strength_top_k
            ),
            "full_set_candidate_pool_query_match_strength_top_k": int(
                args.full_set_candidate_pool_query_match_strength_top_k
            ),
            "full_set_candidate_pool_query_top_k": int(args.full_set_candidate_pool_query_top_k),
            "full_set_candidate_pool_query_cell_top_k": int(args.full_set_candidate_pool_query_cell_top_k),
            "full_set_candidate_pool_query_depth_top_k": int(args.full_set_candidate_pool_query_depth_top_k),
            "full_set_source_anchor_idx": None
            if args.full_set_source_anchor_idx is None
            else str(Path(args.full_set_source_anchor_idx)),
            "full_set_target_query_support": float(args.full_set_target_query_support),
            "full_set_min_visibility": float(args.full_set_min_visibility),
            "full_set_min_mask_validity": float(args.full_set_min_mask_validity),
            "full_set_max_per_spatial_cluster": int(args.full_set_max_per_spatial_cluster),
            "full_set_visibility_weight": float(args.full_set_visibility_weight),
            "full_set_mask_weight": float(args.full_set_mask_weight),
            "full_set_solver_weight": float(args.full_set_solver_weight),
            "full_set_query_support_weight": float(args.full_set_query_support_weight),
            "full_set_geometry_weight": float(args.full_set_geometry_weight),
            "full_set_depth_weight": float(args.full_set_depth_weight),
            "full_set_query_geometry_weight": float(args.full_set_query_geometry_weight),
            "full_set_query_depth_weight": float(args.full_set_query_depth_weight),
            "full_set_query_bearing_weight": float(args.full_set_query_bearing_weight),
            "full_set_query_pnp_geometry_weight": float(args.full_set_query_pnp_geometry_weight),
            "full_set_query_pnp_distance_scale_m": float(args.full_set_query_pnp_distance_scale_m),
            "full_set_support_match_strength": None
            if args.full_set_support_match_strength is None
            else str(Path(args.full_set_support_match_strength)),
            "full_set_match_competition_risk": None
            if args.full_set_match_competition_risk is None
            else str(Path(args.full_set_match_competition_risk)),
            "full_set_support_match_strength_weight": float(args.full_set_support_match_strength_weight),
            "full_set_match_competition_risk_weight": float(args.full_set_match_competition_risk_weight),
            "full_set_match_competition_risk_reject_threshold": float(
                args.full_set_match_competition_risk_reject_threshold
            ),
            "full_set_query_match_dominance": None
            if args.full_set_query_match_dominance is None
            else str(Path(args.full_set_query_match_dominance)),
            "full_set_query_match_strength_weight": float(args.full_set_query_match_strength_weight),
            "full_set_query_match_competition_weight": float(args.full_set_query_match_competition_weight),
            "full_set_query_match_competition_ratio_weight": float(
                args.full_set_query_match_competition_ratio_weight
            ),
            "full_set_query_match_reference_sampled_idx": None
            if args.full_set_query_match_reference_sampled_idx is None
            else str(Path(args.full_set_query_match_reference_sampled_idx)),
            "full_set_query_match_risk_reference_margin": float(
                args.full_set_query_match_risk_reference_margin
            ),
            "full_set_query_match_risk_reference_weight": float(
                args.full_set_query_match_risk_reference_weight
            ),
            "full_set_query_match_nonreference_competition_weight": float(
                args.full_set_query_match_nonreference_competition_weight
            ),
            "full_set_query_prefill_target_fraction": float(args.full_set_query_prefill_target_fraction),
            "full_set_query_prefill_max_candidates_per_query": int(args.full_set_query_prefill_max_candidates_per_query),
            "full_set_query_prefill_max_per_spatial_cluster": int(args.full_set_query_prefill_max_per_spatial_cluster),
            "full_set_query_prefill_min_cells_per_query": int(args.full_set_query_prefill_min_cells_per_query),
            "full_set_query_prefill_min_depth_bins_per_query": int(args.full_set_query_prefill_min_depth_bins_per_query),
            "full_set_query_prefill_hard_coverage_bonus": float(args.full_set_query_prefill_hard_coverage_bonus),
            "full_set_source_query_prefill_target_fraction": float(
                args.full_set_source_query_prefill_target_fraction
            ),
            "full_set_source_query_prefill_max_candidates_per_query": int(
                args.full_set_source_query_prefill_max_candidates_per_query
            ),
            "full_set_source_query_prefill_max_per_spatial_cluster": int(
                args.full_set_source_query_prefill_max_per_spatial_cluster
            ),
            "full_set_source_query_prefill_min_cells_per_query": int(
                args.full_set_source_query_prefill_min_cells_per_query
            ),
            "full_set_source_query_prefill_min_depth_bins_per_query": int(
                args.full_set_source_query_prefill_min_depth_bins_per_query
            ),
            "full_set_validation_query_prefill_target_fraction": float(
                args.full_set_validation_query_prefill_target_fraction
            ),
            "full_set_validation_query_prefill_max_candidates_per_query": int(
                args.full_set_validation_query_prefill_max_candidates_per_query
            ),
            "full_set_validation_query_prefill_max_per_spatial_cluster": int(
                args.full_set_validation_query_prefill_max_per_spatial_cluster
            ),
            "full_set_validation_query_prefill_min_cells_per_query": int(
                args.full_set_validation_query_prefill_min_cells_per_query
            ),
            "full_set_validation_query_prefill_min_depth_bins_per_query": int(
                args.full_set_validation_query_prefill_min_depth_bins_per_query
            ),
            "full_set_validation_query_prefill_force_all": bool(
                args.full_set_validation_query_prefill_force_all
            ),
            "full_set_main_min_query_gain": float(args.full_set_main_min_query_gain),
            "full_set_main_dynamic_lookahead": int(args.full_set_main_dynamic_lookahead),
            "full_set_kc_anchor_count": int(args.full_set_kc_anchor_count),
            "full_set_kc_anchor_min_score": float(args.full_set_kc_anchor_min_score),
            "full_set_kc_anchor_max_per_spatial_cluster": int(args.full_set_kc_anchor_max_per_spatial_cluster),
            "full_set_source_anchor_mode": str(args.full_set_source_anchor_mode),
            "full_set_source_anchor_weight": float(args.full_set_source_anchor_weight),
            "full_set_force_source_anchor": bool(args.full_set_force_source_anchor),
            "full_set_descriptor_conflict_graph": None
            if args.full_set_descriptor_conflict_graph is None
            else str(Path(args.full_set_descriptor_conflict_graph)),
            "full_set_descriptor_conflict_weight": float(args.full_set_descriptor_conflict_weight),
            "full_set_auto_descriptor_conflict_top_k": int(args.full_set_auto_descriptor_conflict_top_k),
            "full_set_auto_descriptor_conflict_candidate_limit": int(
                args.full_set_auto_descriptor_conflict_candidate_limit
            ),
            "full_set_auto_descriptor_conflict_min_cosine": float(
                args.full_set_auto_descriptor_conflict_min_cosine
            ),
            "full_set_auto_descriptor_conflict_min_spatial_distance_m": float(
                args.full_set_auto_descriptor_conflict_min_spatial_distance_m
            ),
            "full_set_sparse_validation_profile": None
            if args.full_set_sparse_validation_profile is None
            else str(Path(args.full_set_sparse_validation_profile)),
            "full_set_allow_metric_only_sparse_validation_profile": bool(
                args.full_set_allow_metric_only_sparse_validation_profile
            ),
            "full_set_sparse_validation_support_scope": str(
                args.full_set_sparse_validation_support_scope
            ),
            "full_set_sparse_validation_risk_weight": float(args.full_set_sparse_validation_risk_weight),
            "full_set_sparse_validation_risk_reject_threshold": float(
                args.full_set_sparse_validation_risk_reject_threshold
            ),
            "full_set_sparse_validation_protected_hard_reject_exempt": bool(
                args.full_set_sparse_validation_protected_hard_reject_exempt
            ),
            "full_set_sparse_validation_source_anchor_risk_exempt": bool(
                args.full_set_sparse_validation_source_anchor_risk_exempt
            ),
            "full_set_sparse_validation_risk_expand_top_k": int(
                args.full_set_sparse_validation_risk_expand_top_k
            ),
            "full_set_sparse_validation_risk_expand_neighbors_per_seed": int(
                args.full_set_sparse_validation_risk_expand_neighbors_per_seed
            ),
            "full_set_sparse_validation_risk_expand_candidate_limit": int(
                args.full_set_sparse_validation_risk_expand_candidate_limit
            ),
            "full_set_sparse_validation_risk_expand_min_cosine": float(
                args.full_set_sparse_validation_risk_expand_min_cosine
            ),
            "full_set_sparse_validation_risk_expand_min_spatial_distance_m": float(
                args.full_set_sparse_validation_risk_expand_min_spatial_distance_m
            ),
            "full_set_sparse_validation_risk_expand_weight": float(
                args.full_set_sparse_validation_risk_expand_weight
            ),
            "full_set_sparse_validation_conflict_max_landmarks_per_query": int(
                args.full_set_sparse_validation_conflict_max_landmarks_per_query
            ),
            "full_set_final_prune_no_query_utility": bool(args.full_set_final_prune_no_query_utility),
            "full_set_final_prune_conflict_threshold": float(args.full_set_final_prune_conflict_threshold),
            "full_set_final_prune_min_keep_count": int(args.full_set_final_prune_min_keep_count),
            "full_set_final_prune_query_match_risk_reference": bool(
                args.full_set_final_prune_query_match_risk_reference
            ),
            "full_set_final_prune_nonreference_query_match_competition": bool(
                args.full_set_final_prune_nonreference_query_match_competition
            ),
            "full_set_precision_fill_target_count": int(args.full_set_precision_fill_target_count),
            "full_set_precision_fill_min_score": float(args.full_set_precision_fill_min_score),
            "full_set_precision_fill_kc_weight": float(args.full_set_precision_fill_kc_weight),
            "full_set_precision_fill_visibility_weight": float(
                args.full_set_precision_fill_visibility_weight
            ),
            "full_set_precision_fill_mask_weight": float(args.full_set_precision_fill_mask_weight),
            "full_set_precision_fill_solver_weight": float(args.full_set_precision_fill_solver_weight),
            "full_set_precision_fill_query_support_weight": float(
                args.full_set_precision_fill_query_support_weight
            ),
            "full_set_precision_fill_support_match_strength_weight": float(
                args.full_set_precision_fill_support_match_strength_weight
            ),
            "full_set_precision_fill_ambiguity_risk_weight": float(
                args.full_set_precision_fill_ambiguity_risk_weight
            ),
            "full_set_precision_fill_sparse_validation_risk_weight": float(
                args.full_set_precision_fill_sparse_validation_risk_weight
            ),
            "full_set_precision_fill_match_competition_risk_weight": float(
                args.full_set_precision_fill_match_competition_risk_weight
            ),
            "full_set_precision_fill_descriptor_conflict_weight": float(
                args.full_set_precision_fill_descriptor_conflict_weight
            ),
            "full_set_precision_fill_require_query_support": bool(
                args.full_set_precision_fill_require_query_support
            ),
            "full_set_precision_fill_max_per_spatial_cluster": int(
                args.full_set_precision_fill_max_per_spatial_cluster
            ),
            "full_set_preserve_source_descriptors": bool(args.full_set_preserve_source_descriptors),
            "full_set_validation_min_query_landmarks": int(args.full_set_validation_min_query_landmarks),
            "full_set_validation_min_query_cells": int(args.full_set_validation_min_query_cells),
            "full_set_validation_min_query_depth_bins": int(args.full_set_validation_min_query_depth_bins),
            "full_set_validation_min_query_bearing_spread": float(
                args.full_set_validation_min_query_bearing_spread
            ),
            "full_set_validation_min_query_camera_spread_m": float(
                args.full_set_validation_min_query_camera_spread_m
            ),
            "full_set_local_search_rounds": int(args.full_set_local_search_rounds),
            "full_set_local_search_candidates_per_query": int(args.full_set_local_search_candidates_per_query),
            "full_set_local_search_max_additions": int(args.full_set_local_search_max_additions),
            "full_set_local_search_reserve_count": int(args.full_set_local_search_reserve_count),
            "full_set_local_prune_conflict_threshold": float(args.full_set_local_prune_conflict_threshold),
            "full_set_local_prune_validate_queries": bool(args.full_set_local_prune_validate_queries),
            "full_set_augmentation_min_solver_support": float(args.full_set_augmentation_min_solver_support),
            "full_set_augmentation_min_query_support": float(args.full_set_augmentation_min_query_support),
            "full_set_augmentation_min_kc_score": float(args.full_set_augmentation_min_kc_score),
            "require_candidate_gain": bool(args.require_candidate_gain),
        },
        "branch_selection": False,
        "single_path_deployment": True,
        "fixed_native_geometry": True,
        "metrics_summary": metrics,
        "split_audit": split_audit,
    }
    if native_safe_core_summary is not None:
        metrics["native_safe_core"] = native_safe_core_summary
        manifest["native_safe_core"] = native_safe_core_summary
    if full_sparse_set_summary is not None:
        metrics["full_gaussian_sparse_set"] = full_sparse_set_summary
        manifest["full_gaussian_sparse_set"] = full_sparse_set_summary
    if reused_sampled_idx_summary is not None:
        metrics["reused_sampled_idx"] = reused_sampled_idx_summary
        manifest["reused_sampled_idx"] = reused_sampled_idx_summary
    write_map_resample_audit_bundle(output_eval_dir, manifest, metrics, split_audit)
    (output_eval_dir / "command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_eval_dir / "map_command.txt").write_text(manifest["command"] + "\n", encoding="utf-8")
    (output_eval_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    (output_eval_dir / "ulf_git_status.txt").write_text(_git_status_at(ulf_root), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
