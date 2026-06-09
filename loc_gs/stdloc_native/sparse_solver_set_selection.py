from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


_SOLVER_SUPPORT_FIELDS = (
    "support",
    "viable_tuple_mass",
    "logdet_H",
    "min_eigenvalue",
)


@dataclass(frozen=True)
class SparseSetSignals:
    """Full-Gaussian sparse selection signals.

    All dense vectors are indexed by the original Gaussian id.  Per-query support
    maps use the same ids and are the only query-level solver evidence consumed
    by the sparse-only selector.
    """

    xyz: torch.Tensor
    kc_score: torch.Tensor
    visibility_score: torch.Tensor
    mask_validity: torch.Tensor
    solver_support: torch.Tensor
    per_query_support: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    validation_per_query_support: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    per_query_match_strength: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    per_query_match_competition_risk: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    per_query_match_competition_risk_reference: Mapping[str, float] = field(default_factory=dict)
    per_query_match_competition_reference_landmarks: Mapping[str, object] = field(default_factory=dict)
    per_query_observations: Mapping[str, Mapping[int, Mapping[str, object]]] = field(default_factory=dict)
    descriptor_conflict_edges: Mapping[int, Mapping[int, float]] = field(default_factory=dict)
    descriptor_features: torch.Tensor | None = None
    ambiguity_risk: torch.Tensor | None = None
    sparse_validation_risk: torch.Tensor | None = None
    sparse_validation_hard_reject_exempt: torch.Tensor | None = None
    support_match_strength: torch.Tensor | None = None
    match_competition_risk: torch.Tensor | None = None
    source_anchor_idx: torch.Tensor | None = None


@dataclass(frozen=True)
class SparseSetSelectionConfig:
    max_landmarks: int | None = None
    min_landmarks: int = 0
    min_marginal_gain: float = 0.05
    candidate_pool_size: int | None = None
    candidate_pool_kc_top_k: int = 0
    candidate_pool_visibility_top_k: int = 0
    candidate_pool_mask_top_k: int = 0
    candidate_pool_solver_top_k: int = 0
    candidate_pool_match_strength_top_k: int = 0
    candidate_pool_query_match_strength_top_k: int = 0
    candidate_pool_query_top_k: int = 0
    candidate_pool_query_cell_top_k: int = 0
    candidate_pool_query_depth_top_k: int = 0
    target_query_support: float = 64.0
    spatial_cluster_size_m: float = 0.5
    max_per_spatial_cluster: int = 0
    depth_bin_size_m: float = 2.0
    min_visibility: float = 0.0
    min_mask_validity: float = 0.0
    kc_weight: float = 1.0
    visibility_weight: float = 0.5
    mask_weight: float = 0.5
    solver_weight: float = 1.0
    query_support_weight: float = 2.0
    geometry_weight: float = 0.5
    depth_weight: float = 0.25
    query_geometry_weight: float = 0.0
    query_depth_weight: float = 0.0
    query_bearing_weight: float = 0.0
    query_pnp_geometry_weight: float = 0.0
    query_pnp_distance_scale_m: float = 2.0
    ambiguity_risk_weight: float = 1.0
    sparse_validation_risk_weight: float = 0.0
    sparse_validation_risk_reject_threshold: float = 0.0
    sparse_validation_risk_protected_hard_reject_exempt: bool = False
    sparse_validation_risk_source_anchor_exempt: bool = False
    support_match_strength_weight: float = 0.0
    match_competition_risk_weight: float = 0.0
    match_competition_risk_reject_threshold: float = 0.0
    query_match_strength_weight: float = 0.0
    query_match_competition_weight: float = 0.0
    query_match_competition_ratio_weight: float = 0.0
    query_match_risk_reference_margin: float = 0.0
    query_match_risk_reference_weight: float = 0.0
    query_match_nonreference_competition_weight: float = 0.0
    sparse_validation_risk_expand_top_k: int = 0
    sparse_validation_risk_expand_neighbors_per_seed: int = 16
    sparse_validation_risk_expand_candidate_limit: int = 65536
    sparse_validation_risk_expand_min_cosine: float = 0.98
    sparse_validation_risk_expand_min_spatial_distance_m: float = 1.0
    sparse_validation_risk_expand_weight: float = 1.0
    descriptor_conflict_weight: float = 0.0
    auto_descriptor_conflict_top_k: int = 0
    auto_descriptor_conflict_candidate_limit: int = 8192
    auto_descriptor_conflict_min_cosine: float = 0.95
    auto_descriptor_conflict_min_spatial_distance_m: float = 1.0
    easy_query_gain_decay: float = 0.05
    validation_min_query_landmarks: int = 0
    validation_min_query_cells: int = 0
    validation_min_query_depth_bins: int = 0
    validation_min_query_bearing_spread: float = 0.0
    validation_min_query_camera_spread_m: float = 0.0
    local_search_rounds: int = 0
    local_search_candidates_per_query: int = 0
    local_search_max_additions: int = 0
    local_search_reserve_count: int = 0
    local_prune_conflict_threshold: float = 0.0
    local_prune_validate_queries: bool = False
    query_prefill_target_fraction: float = 0.0
    query_prefill_max_candidates_per_query: int = 0
    query_prefill_max_per_spatial_cluster: int = 0
    query_prefill_min_cells_per_query: int = 0
    query_prefill_min_depth_bins_per_query: int = 0
    query_prefill_hard_coverage_bonus: float = 1.0e6
    source_query_prefill_target_fraction: float = 0.0
    source_query_prefill_max_candidates_per_query: int = 0
    source_query_prefill_max_per_spatial_cluster: int = 0
    source_query_prefill_min_cells_per_query: int = 0
    source_query_prefill_min_depth_bins_per_query: int = 0
    validation_query_prefill_target_fraction: float = 0.0
    validation_query_prefill_max_candidates_per_query: int = 0
    validation_query_prefill_max_per_spatial_cluster: int = 0
    validation_query_prefill_min_cells_per_query: int = 0
    validation_query_prefill_min_depth_bins_per_query: int = 0
    validation_query_prefill_force_all: bool = False
    main_min_query_gain: float = 0.0
    main_dynamic_lookahead: int = 0
    kc_anchor_count: int = 0
    kc_anchor_min_score: float = 0.0
    kc_anchor_max_per_spatial_cluster: int = 0
    source_anchor_mode: str = "hard"
    source_anchor_weight: float = 0.0
    force_source_anchor: bool = False
    final_prune_no_query_utility: bool = False
    final_prune_conflict_threshold: float = 0.0
    final_prune_min_keep_count: int = 0
    final_prune_query_match_risk_reference: bool = False
    final_prune_nonreference_query_match_competition: bool = False
    precision_fill_target_count: int = 0
    precision_fill_min_score: float = 0.0
    precision_fill_kc_weight: float = 0.0
    precision_fill_visibility_weight: float = 0.0
    precision_fill_mask_weight: float = 0.0
    precision_fill_solver_weight: float = 0.0
    precision_fill_query_support_weight: float = 0.0
    precision_fill_support_match_strength_weight: float = 0.0
    precision_fill_ambiguity_risk_weight: float = 0.0
    precision_fill_sparse_validation_risk_weight: float = 0.0
    precision_fill_match_competition_risk_weight: float = 0.0
    precision_fill_descriptor_conflict_weight: float = 0.0
    precision_fill_require_query_support: bool = False
    precision_fill_max_per_spatial_cluster: int = 0
    augmentation_min_solver_support: float = 0.0
    augmentation_min_query_support: float = 0.0
    augmentation_min_kc_score: float = 0.0


@dataclass(frozen=True)
class SparseSetSelectionResult:
    sampled_idx: torch.Tensor
    metadata: dict


@dataclass(frozen=True)
class QueryObservation:
    cell: tuple[int, int, int] | None = None
    depth: int | None = None
    xy_norm: tuple[float, float] | None = None
    bearing: tuple[float, float, float] | None = None
    camera_xyz: tuple[float, float, float] | None = None
    depth_m: float | None = None
    reprojection_error_px: float | None = None
    descriptor_margin: float | None = None
    local_geometry_score: float | None = None
    ray_artifact_score: float | None = None
    ray_entropy: float | None = None


def _metric_value(metrics: Mapping[str, object]) -> float:
    value = 0.0
    for key in _SOLVER_SUPPORT_FIELDS:
        if key in metrics:
            value += max(0.0, float(metrics[key]))  # type: ignore[arg-type]
    return float(value)


def per_query_support_from_solver_constraints(payload: Mapping[str, object]) -> dict[str, dict[int, float]]:
    """Convert old edit-style solver constraints into set-selection evidence.

    ``candidate_gain`` and ``source_loss`` were named for replacement edits.  In
    full-set selection both mean that a Gaussian has positive sparse solver
    evidence for a query, so both tables are merged into a query -> landmark map.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("solver constraints must be a mapping")
    out: dict[str, dict[int, float]] = {}
    for table_name in ("candidate_gain", "source_loss"):
        raw_table = payload.get(table_name, {})
        if not isinstance(raw_table, Mapping):
            continue
        for raw_gid, raw_query_map in raw_table.items():
            gid = int(raw_gid)
            if not isinstance(raw_query_map, Mapping):
                continue
            for raw_query_id, raw_metrics in raw_query_map.items():
                if not isinstance(raw_metrics, Mapping):
                    continue
                score = _metric_value(raw_metrics)
                if score <= 0.0:
                    continue
                query_id = str(raw_query_id)
                out.setdefault(query_id, {})
                out[query_id][gid] = float(out[query_id].get(gid, 0.0) + score)
    return out


def _load_sparse_validation_profile_payload(profile: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(profile, (str, Path)):
        path = Path(profile)
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = profile
    if not isinstance(payload, Mapping):
        raise TypeError("sparse PnP validation profile must be a mapping or JSON path")
    split = str(payload.get("split_name", "")).strip().lower()
    if split == "test":
        raise ValueError("test split sparse PnP validation profile is not allowed")
    for metadata_key in ("feedback_metadata", "protected_feedback_metadata", "validated_feedback_metadata"):
        metadata = payload.get(metadata_key, {})
        if not isinstance(metadata, Mapping):
            continue
        for split_key in (
            "split_name",
            "feedback_bank_split_name",
            "candidate_feedback_bank_split_name",
            "baseline_feedback_bank_split_name",
            "validated_feedback_bank_split_name",
        ):
            if str(metadata.get(split_key, "")).strip().lower() == "test":
                raise ValueError("test split sparse PnP validation profile is not allowed")
    return payload


def _profile_query_support_table(
    raw: object,
    *,
    num_gaussians: int,
    prefix: str,
    allowed_query_ids: set[str] | None = None,
) -> tuple[dict[str, dict[int, float]], set[int]]:
    out: dict[str, dict[int, float]] = {}
    landmark_ids: set[int] = set()
    if not isinstance(raw, Mapping):
        return out, landmark_ids
    for raw_query_id, raw_query_map in raw.items():
        if not isinstance(raw_query_map, Mapping):
            continue
        raw_query_key = str(raw_query_id)
        if allowed_query_ids is not None and raw_query_key not in allowed_query_ids:
            continue
        query_id = f"{prefix}:{raw_query_key}"
        for raw_gid, raw_value in raw_query_map.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= gid < int(num_gaussians) and math.isfinite(value) and value > 0.0:
                out.setdefault(query_id, {})
                out[query_id][gid] = float(out[query_id].get(gid, 0.0) + value)
                landmark_ids.add(gid)
    return out, landmark_ids


def _profile_query_observation_table(
    raw: object,
    *,
    num_gaussians: int,
    prefix: str,
    allowed_query_ids: set[str] | None = None,
) -> dict[str, dict[int, dict[str, object]]]:
    out: dict[str, dict[int, dict[str, object]]] = {}
    if not isinstance(raw, Mapping):
        return out
    for raw_query_id, raw_query_map in raw.items():
        if not isinstance(raw_query_map, Mapping):
            continue
        raw_query_key = str(raw_query_id)
        if allowed_query_ids is not None and raw_query_key not in allowed_query_ids:
            continue
        query_id = f"{prefix}:{raw_query_key}"
        for raw_gid, raw_value in raw_query_map.items():
            try:
                gid = int(raw_gid)
            except (TypeError, ValueError):
                continue
            if 0 <= gid < int(num_gaussians) and isinstance(raw_value, Mapping):
                out.setdefault(query_id, {})[gid] = dict(raw_value)
    return out


def sparse_validation_signals_from_pnp_profile(
    profile: Mapping[str, Any] | str | Path,
    *,
    num_gaussians: int,
    support_scope: str = "all",
    require_attribution: bool = False,
) -> dict[str, Any]:
    """Convert a sparse PnP validation profile into selector-side signals.

    This keeps the validation loop explicit: candidate-induced regression becomes
    a landmark risk vector, while protected baseline support and validated hard
    improvements become per-query support that the set optimizer can preserve.
    """

    size = int(num_gaussians)
    if size <= 0:
        raise ValueError("num_gaussians must be positive")
    payload = _load_sparse_validation_profile_payload(profile)
    attribution_status = str(payload.get("attribution_status", "unknown")).strip() or "unknown"
    if bool(require_attribution) and attribution_status != "attributed":
        raise ValueError(
            "sparse PnP validation profile must have attribution_status='attributed' "
            f"for main SparseSet selection; got {attribution_status!r}"
        )
    scope = str(support_scope).strip().lower()
    if scope not in {"all", "protected_regressions"}:
        raise ValueError(f"unsupported sparse validation support scope: {support_scope!r}")
    protected_allowed_query_ids = None
    if scope == "protected_regressions":
        protected_allowed_query_ids = {
            str(query_id)
            for query_id in payload.get("protected_regression_query_ids", [])
            if str(query_id).strip()
        }

    risk = torch.zeros(size, dtype=torch.float32)
    raw_risk = payload.get("landmark_regression_risk", {})
    if isinstance(raw_risk, Mapping):
        for raw_gid, raw_value in raw_risk.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if 0 <= gid < size and math.isfinite(value) and value > 0.0:
                risk[gid] = max(float(risk[gid].item()), value)

    protected_support, protected_ids = _profile_query_support_table(
        payload.get("protected_per_query_support", {}),
        num_gaussians=size,
        prefix="protected",
        allowed_query_ids=protected_allowed_query_ids,
    )
    validated_support, validated_ids = _profile_query_support_table(
        payload.get("validated_per_query_support", {}),
        num_gaussians=size,
        prefix="validated",
    )
    validation_support: dict[str, dict[int, float]] = {}
    validation_support.update(protected_support)
    validation_support.update(validated_support)

    protected_observations = _profile_query_observation_table(
        payload.get("protected_per_query_observations", {}),
        num_gaussians=size,
        prefix="protected",
        allowed_query_ids=protected_allowed_query_ids,
    )
    validated_observations = _profile_query_observation_table(
        payload.get("validated_per_query_observations", {}),
        num_gaussians=size,
        prefix="validated",
    )
    observations: dict[str, dict[int, dict[str, object]]] = {}
    observations.update(protected_observations)
    observations.update(validated_observations)

    exempt = torch.zeros(size, dtype=torch.bool)
    for gid in sorted(protected_ids | validated_ids):
        exempt[int(gid)] = True

    return {
        "sparse_validation_risk": risk,
        "sparse_validation_hard_reject_exempt": exempt,
        "validation_per_query_support": validation_support,
        "per_query_observations": observations,
        "metadata": {
            "source": "sparse_pnp_validation_profile",
            "attribution_status": attribution_status,
            "split_name": str(payload.get("split_name", "unknown")),
            "scene": str(payload.get("scene", "")),
            "support_scope": str(scope),
            "risk_landmark_count": int((risk > 0).sum().item()),
            "protected_support_query_count": int(len(protected_support)),
            "validated_support_query_count": int(len(validated_support)),
            "protected_support_landmark_count": int(len(protected_ids)),
            "validated_support_landmark_count": int(len(validated_ids)),
            "hard_reject_exempt_landmark_count": int(exempt.sum().item()),
        },
    }


def _float_vector(values: torch.Tensor, *, name: str, size: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if size is not None and int(tensor.numel()) != int(size):
        raise ValueError(f"{name} length {tensor.numel()} does not match xyz length {size}")
    return tensor


def _bool_vector(values: torch.Tensor, *, name: str, size: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.bool).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if size is not None and int(tensor.numel()) != int(size):
        raise ValueError(f"{name} length {tensor.numel()} does not match xyz length {size}")
    return tensor


def _feature_matrix(values: torch.Tensor | None, *, name: str, size: int) -> torch.Tensor | None:
    if values is None:
        return None
    tensor = torch.as_tensor(values, dtype=torch.float32).cpu()
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [N, C]")
    if int(tensor.shape[0]) != int(size):
        raise ValueError(f"{name} row count {tensor.shape[0]} does not match xyz length {size}")
    if int(tensor.shape[1]) <= 0:
        raise ValueError(f"{name} must have at least one channel")
    return tensor


def _xyz_tensor(values: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).cpu()
    if tensor.ndim != 2 or tensor.shape[1] != 3 or tensor.shape[0] == 0:
        raise ValueError("xyz must have shape [N, 3]")
    return tensor


def _normalize(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    out = torch.zeros_like(values, dtype=torch.float32)
    if not bool(finite.any()):
        return out
    valid = values[finite].float()
    lo = float(valid.min().item())
    hi = float(valid.max().item())
    if not math.isfinite(lo) or not math.isfinite(hi):
        return out
    if hi <= lo:
        out[finite] = 1.0 if hi > 0.0 else 0.0
    else:
        out[finite] = ((values[finite] - lo) / (hi - lo)).clamp(0.0, 1.0)
    return out


def _signal_stats_for_ids(values: torch.Tensor, ids: list[int] | set[int] | torch.Tensor) -> dict[str, float | int | None]:
    raw_ids = torch.as_tensor(list(ids) if not isinstance(ids, torch.Tensor) else ids, dtype=torch.long).reshape(-1).cpu()
    if int(raw_ids.numel()) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "nonzero_count": 0}
    valid_ids = raw_ids[(raw_ids >= 0) & (raw_ids < int(values.numel()))]
    if int(valid_ids.numel()) == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "nonzero_count": 0}
    selected_values = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()[valid_ids]
    finite = selected_values[torch.isfinite(selected_values)]
    if int(finite.numel()) == 0:
        return {
            "count": int(valid_ids.numel()),
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "nonzero_count": 0,
        }
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "median": float(torch.quantile(finite, 0.5).item()),
        "p10": float(torch.quantile(finite, 0.1).item()),
        "p90": float(torch.quantile(finite, 0.9).item()),
        "nonzero_count": int((finite > 0.0).sum().item()),
    }


def _cluster_key(point: torch.Tensor, grid_size: float) -> tuple[int, int, int]:
    size = max(1.0e-6, float(grid_size))
    values = torch.floor(point.float() / size).to(torch.long).tolist()
    return int(values[0]), int(values[1]), int(values[2])


def _depth_bin(point: torch.Tensor, bin_size: float) -> int:
    size = max(1.0e-6, float(bin_size))
    return int(math.floor(float(point[2].item()) / size))


def _canonical_query_support(
    per_query_support: Mapping[str, Mapping[int, float]],
    *,
    size: int,
) -> tuple[dict[int, dict[str, float]], tuple[str, ...]]:
    by_landmark: dict[int, dict[str, float]] = {}
    query_ids: list[str] = []
    for raw_query_id, raw_landmarks in per_query_support.items():
        query_id = str(raw_query_id)
        query_ids.append(query_id)
        for raw_gid, raw_value in raw_landmarks.items():
            gid = int(raw_gid)
            if gid < 0 or gid >= size:
                continue
            value = max(0.0, float(raw_value))
            if value <= 0.0:
                continue
            by_landmark.setdefault(gid, {})[query_id] = value
    return by_landmark, tuple(query_ids)


def _query_table_by_query(
    by_landmark: Mapping[int, Mapping[str, float]],
) -> dict[str, list[tuple[int, float]]]:
    by_query: dict[str, list[tuple[int, float]]] = {}
    for raw_gid, query_map in by_landmark.items():
        gid = int(raw_gid)
        for raw_query_id, raw_value in query_map.items():
            value = max(0.0, float(raw_value))
            if value <= 0.0:
                continue
            by_query.setdefault(str(raw_query_id), []).append((gid, value))
    return by_query


def _canonical_query_scalar_table(values: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_query_id, raw_value in values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value < 0.0:
            continue
        out[str(raw_query_id)] = float(value)
    return out


def _canonical_query_landmark_sets(
    values: Mapping[str, object],
    *,
    size: int,
) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for raw_query_id, raw_values in values.items():
        query_id = str(raw_query_id)
        if isinstance(raw_values, Mapping):
            raw_iterable = raw_values.keys()
        elif isinstance(raw_values, (set, list, tuple)):
            raw_iterable = raw_values
        else:
            continue
        for raw_gid in raw_iterable:
            try:
                gid = int(raw_gid)
            except (TypeError, ValueError):
                continue
            if gid < 0 or gid >= int(size):
                continue
            out.setdefault(query_id, set()).add(gid)
    return out


def _parse_observation_cell(raw_observation: Mapping[str, object]) -> tuple[int, int, int] | None:
    raw_cell = raw_observation.get("cell")
    if isinstance(raw_cell, (list, tuple)) and len(raw_cell) >= 2:
        try:
            z_value = int(raw_cell[2]) if len(raw_cell) >= 3 else 0
            return int(raw_cell[0]), int(raw_cell[1]), z_value
        except (TypeError, ValueError):
            return None
    key_pairs = (("cell_x", "cell_y"), ("grid_x", "grid_y"), ("image_cell_x", "image_cell_y"))
    for x_key, y_key in key_pairs:
        if x_key in raw_observation and y_key in raw_observation:
            try:
                return int(raw_observation[x_key]), int(raw_observation[y_key]), int(raw_observation.get("cell_z", 0))
            except (TypeError, ValueError):
                return None
    return None


def _parse_observation_depth(raw_observation: Mapping[str, object]) -> int | None:
    for key in ("depth_bin", "query_depth_bin"):
        if key in raw_observation:
            try:
                return int(raw_observation[key])
            except (TypeError, ValueError):
                return None
    for key in ("depth", "depth_m", "query_depth"):
        if key in raw_observation:
            try:
                return int(math.floor(float(raw_observation[key])))
            except (TypeError, ValueError):
                return None
    return None


def _parse_scalar(raw_observation: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in raw_observation:
            continue
        try:
            value = float(raw_observation[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return float(value)
    return None


def _parse_vector2(raw_observation: Mapping[str, object], keys: tuple[str, ...]) -> tuple[float, float] | None:
    for key in keys:
        raw = raw_observation.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                values = (float(raw[0]), float(raw[1]))
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values):
                return values
    for x_key, y_key in (
        ("x_norm", "y_norm"),
        ("query_x_norm", "query_y_norm"),
        ("u_norm", "v_norm"),
        ("normalized_x", "normalized_y"),
    ):
        if x_key in raw_observation and y_key in raw_observation:
            try:
                values = (float(raw_observation[x_key]), float(raw_observation[y_key]))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values):
                return values
    return None


def _parse_vector3(raw_observation: Mapping[str, object], keys: tuple[str, ...]) -> tuple[float, float, float] | None:
    for key in keys:
        raw = raw_observation.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                values = (float(raw[0]), float(raw[1]), float(raw[2]))
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values):
                return values
    return None


def _parse_observation_bearing(raw_observation: Mapping[str, object]) -> tuple[float, float, float] | None:
    return _parse_vector3(
        raw_observation,
        ("bearing", "bearing_vector", "unit_bearing", "ray", "ray_direction", "query_bearing"),
    )


def _parse_observation_camera_xyz(raw_observation: Mapping[str, object]) -> tuple[float, float, float] | None:
    return _parse_vector3(
        raw_observation,
        ("camera_xyz", "camera_point", "point_cam", "query_camera_xyz", "xyz_cam", "camera_frame_xyz"),
    )


def _canonical_query_observations(
    per_query_observations: Mapping[str, Mapping[int, Mapping[str, object]]],
    *,
    size: int,
) -> dict[int, dict[str, QueryObservation]]:
    by_landmark: dict[int, dict[str, QueryObservation]] = {}
    if not isinstance(per_query_observations, Mapping):
        return by_landmark
    for raw_query_id, raw_landmarks in per_query_observations.items():
        if not isinstance(raw_landmarks, Mapping):
            continue
        query_id = str(raw_query_id)
        for raw_gid, raw_observation in raw_landmarks.items():
            try:
                gid = int(raw_gid)
            except (TypeError, ValueError):
                continue
            if gid < 0 or gid >= int(size) or not isinstance(raw_observation, Mapping):
                continue
            cell = _parse_observation_cell(raw_observation)
            depth = _parse_observation_depth(raw_observation)
            xy_norm = _parse_vector2(
                raw_observation,
                ("xy_norm", "query_xy_norm", "normalized_xy", "query_normalized_xy", "uv_norm", "keypoint_xy_norm"),
            )
            bearing = _parse_observation_bearing(raw_observation)
            camera_xyz = _parse_observation_camera_xyz(raw_observation)
            depth_m = _parse_scalar(raw_observation, ("depth", "depth_m", "expected_depth", "query_depth"))
            reprojection_error_px = _parse_scalar(raw_observation, ("reprojection_error_px", "reproj_error_px"))
            descriptor_margin = _parse_scalar(raw_observation, ("descriptor_margin", "match_margin", "nn_margin"))
            local_geometry_score = _parse_scalar(raw_observation, ("local_geometry_score", "geometry_score", "lgcv_score"))
            ray_artifact_score = _parse_scalar(raw_observation, ("ray_artifact_score", "artifact_score", "artifact_risk"))
            ray_entropy = _parse_scalar(raw_observation, ("ray_entropy", "composition_entropy", "alpha_entropy"))
            if (
                cell is None
                and depth is None
                and xy_norm is None
                and bearing is None
                and camera_xyz is None
                and depth_m is None
                and reprojection_error_px is None
                and descriptor_margin is None
                and local_geometry_score is None
                and ray_artifact_score is None
                and ray_entropy is None
            ):
                continue
            by_landmark.setdefault(gid, {})[query_id] = QueryObservation(
                cell=cell,
                depth=depth,
                xy_norm=xy_norm,
                bearing=bearing,
                camera_xyz=camera_xyz,
                depth_m=depth_m,
                reprojection_error_px=reprojection_error_px,
                descriptor_margin=descriptor_margin,
                local_geometry_score=local_geometry_score,
                ray_artifact_score=ray_artifact_score,
                ray_entropy=ray_entropy,
            )
    return by_landmark


def _canonical_descriptor_conflict_edges(
    raw_edges: Mapping[int, Mapping[int, float]],
    *,
    size: int,
) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    if not isinstance(raw_edges, Mapping):
        return out
    for raw_src, raw_neighbors in raw_edges.items():
        try:
            src = int(raw_src)
        except (TypeError, ValueError):
            continue
        if src < 0 or src >= int(size) or not isinstance(raw_neighbors, Mapping):
            continue
        for raw_dst, raw_value in raw_neighbors.items():
            try:
                dst = int(raw_dst)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if dst < 0 or dst >= int(size) or dst == src or value <= 0.0 or not math.isfinite(value):
                continue
            out.setdefault(src, {})
            out[src][dst] = max(float(out[src].get(dst, 0.0)), float(value))
            out.setdefault(dst, {})
            out[dst][src] = max(float(out[dst].get(src, 0.0)), float(value))
    return out


def _merge_descriptor_conflict_edges(
    *tables: Mapping[int, Mapping[int, float]],
    size: int,
) -> dict[int, dict[int, float]]:
    merged: dict[int, dict[int, float]] = {}
    for table in tables:
        canonical = _canonical_descriptor_conflict_edges(table, size=size)
        for src, neighbors in canonical.items():
            for dst, value in neighbors.items():
                merged.setdefault(int(src), {})
                merged[int(src)][int(dst)] = max(float(merged[int(src)].get(int(dst), 0.0)), float(value))
    return merged


def _build_auto_descriptor_conflict_edges(
    descriptor_features: torch.Tensor | None,
    xyz: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    top_k: int,
    candidate_limit: int,
    min_cosine: float,
    min_spatial_distance_m: float,
) -> dict[int, dict[int, float]]:
    """Build a descriptor ambiguity graph from the active candidate pool.

    Edges connect visually similar but spatially distant landmarks.  This is a
    sparse-set selection signal, not a post-hoc branch selector: it uses only
    fixed map descriptors/geometry and is independent of eval query outcomes.
    """

    if descriptor_features is None or int(top_k) <= 0 or int(candidate_ids.numel()) <= 1:
        return {}
    limit = max(0, int(candidate_limit))
    ids = torch.as_tensor(candidate_ids, dtype=torch.long).reshape(-1).cpu()
    if limit > 0 and int(ids.numel()) > limit:
        ids = ids[:limit]
    if int(ids.numel()) <= 1:
        return {}

    feats = F.normalize(descriptor_features[ids].float(), dim=1)
    points = xyz[ids].float()
    k = min(max(1, int(top_k)), int(ids.numel()) - 1)
    sim = feats @ feats.T
    sim.fill_diagonal_(-float("inf"))
    values, positions = torch.topk(sim, k=k, dim=1, largest=True)
    out: dict[int, dict[int, float]] = {}
    cosine_floor = float(min_cosine)
    distance_floor = max(0.0, float(min_spatial_distance_m))
    for row in range(int(ids.numel())):
        src = int(ids[row].item())
        for col in range(k):
            dst_pos = int(positions[row, col].item())
            dst = int(ids[dst_pos].item())
            value = float(values[row, col].item())
            if not math.isfinite(value) or value < cosine_floor:
                continue
            distance = float((points[row] - points[dst_pos]).norm().item())
            if distance < distance_floor:
                continue
            # Far-away descriptor twins are more damaging for PnP than nearby
            # duplicates because they can steal top-1 matches into a different
            # pose basin.  Cap the spatial factor to keep weights interpretable.
            spatial_factor = min(8.0, max(1.0, distance / max(distance_floor, 1.0e-6)))
            edge_score = float(value * spatial_factor)
            out.setdefault(src, {})[dst] = max(float(out.setdefault(src, {}).get(dst, 0.0)), edge_score)
            out.setdefault(dst, {})[src] = max(float(out.setdefault(dst, {}).get(src, 0.0)), edge_score)
    return out


def _expand_sparse_validation_risk_by_descriptor(
    descriptor_features: torch.Tensor | None,
    xyz: torch.Tensor,
    sparse_validation_risk: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    top_k: int,
    neighbors_per_seed: int,
    candidate_limit: int,
    min_cosine: float,
    min_spatial_distance_m: float,
    weight: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Expand observed validation risk to descriptor-twin pose-basin substitutes.

    Sparse PnP regressions are often caused by a wrong pose basin, not by a
    single landmark.  If a risky inlier has far-away descriptor twins in the
    active candidate pool, rejecting only the observed inlier lets the basin
    re-form with substitutes.  This expansion marks those substitutes before
    greedy selection.
    """

    base_risk = torch.as_tensor(sparse_validation_risk, dtype=torch.float32).reshape(-1).cpu()
    metadata: dict[str, float | int] = {
        "sparse_validation_risk_expansion_seed_count": 0,
        "sparse_validation_risk_expansion_candidate_count": 0,
        "sparse_validation_risk_expanded_count": 0,
        "sparse_validation_risk_expansion_edge_count": 0,
        "sparse_validation_risk_expansion_max": 0.0,
    }
    if (
        descriptor_features is None
        or int(top_k) <= 0
        or float(weight) <= 0.0
        or int(candidate_ids.numel()) <= 0
        or int(base_risk.numel()) == 0
    ):
        return base_risk, metadata

    risk_ids = torch.where(base_risk > 0.0)[0]
    if int(risk_ids.numel()) == 0:
        return base_risk, metadata
    seed_count = min(int(top_k), int(risk_ids.numel()))
    _, seed_order = torch.topk(base_risk[risk_ids], k=seed_count, largest=True)
    seed_ids = risk_ids[seed_order].to(torch.long).cpu()

    targets = torch.as_tensor(candidate_ids, dtype=torch.long).reshape(-1).cpu()
    targets = targets[(targets >= 0) & (targets < int(base_risk.numel()))]
    if int(targets.numel()) == 0:
        metadata["sparse_validation_risk_expansion_seed_count"] = int(seed_ids.numel())
        return base_risk, metadata
    limit = max(0, int(candidate_limit))
    if limit > 0 and int(targets.numel()) > limit:
        targets = targets[:limit]

    feats = F.normalize(descriptor_features.float().cpu(), dim=1)
    seed_feats = feats[seed_ids]
    target_feats = feats[targets]
    target_points = xyz[targets].float().cpu()
    out = base_risk.clone()
    cosine_floor = float(min_cosine)
    distance_floor = max(0.0, float(min_spatial_distance_m))
    max_neighbors = int(neighbors_per_seed)
    expansion_edges = 0
    expanded_before = out.clone()

    for seed_row, seed_gid_tensor in enumerate(seed_ids):
        seed_gid = int(seed_gid_tensor.item())
        seed_risk = max(0.0, float(base_risk[seed_gid].item()))
        if seed_risk <= 0.0:
            continue
        similarities = target_feats @ seed_feats[seed_row]
        if max_neighbors > 0 and int(similarities.numel()) > max_neighbors:
            values, positions = torch.topk(similarities, k=max_neighbors, largest=True)
        else:
            values = similarities
            positions = torch.arange(int(similarities.numel()), dtype=torch.long)
        seed_point = xyz[seed_gid].float().cpu()
        for raw_value, raw_position in zip(values.tolist(), positions.tolist()):
            cosine = float(raw_value)
            if not math.isfinite(cosine) or cosine < cosine_floor:
                continue
            target_pos = int(raw_position)
            target_gid = int(targets[target_pos].item())
            if target_gid == seed_gid:
                continue
            distance = float((target_points[target_pos] - seed_point).norm().item())
            if distance < distance_floor:
                continue
            expanded_value = float(seed_risk * cosine * float(weight))
            if expanded_value <= float(out[target_gid].item()):
                continue
            out[target_gid] = expanded_value
            expansion_edges += 1

    expanded_mask = (expanded_before <= 0.0) & (out > 0.0)
    metadata.update(
        {
            "sparse_validation_risk_expansion_seed_count": int(seed_ids.numel()),
            "sparse_validation_risk_expansion_candidate_count": int(targets.numel()),
            "sparse_validation_risk_expanded_count": int(expanded_mask.sum().item()),
            "sparse_validation_risk_expansion_edge_count": int(expansion_edges),
            "sparse_validation_risk_expansion_max": float(out.max().item()) if int(out.numel()) > 0 else 0.0,
        }
    )
    return out, metadata


def _descriptor_conflict_pair_score(
    a: int,
    b: int,
    conflict_edges: Mapping[int, Mapping[int, float]],
) -> float:
    return float(
        max(
            0.0,
            float(conflict_edges.get(int(a), {}).get(int(b), 0.0)),
            float(conflict_edges.get(int(b), {}).get(int(a), 0.0)),
        )
    )


def _descriptor_conflict_penalty(
    gid: int,
    selected_set: set[int],
    conflict_edges: Mapping[int, Mapping[int, float]],
) -> float:
    if not selected_set or not conflict_edges:
        return 0.0
    neighbors = conflict_edges.get(int(gid), {})
    if not neighbors:
        return 0.0
    return float(
        sum(
            max(0.0, float(value))
            for raw_other, value in neighbors.items()
            if int(raw_other) in selected_set
        )
    )


def _query_observation_key(
    gid: int,
    query_id: str,
    observation_by_landmark: Mapping[int, Mapping[str, QueryObservation]],
    *,
    fallback_cluster: tuple[int, int, int],
    fallback_depth_key: int,
) -> tuple[tuple[int, int, int], int]:
    raw = observation_by_landmark.get(int(gid), {}).get(str(query_id))
    if raw is None:
        return fallback_cluster, int(fallback_depth_key)
    return (
        raw.cell if raw.cell is not None else fallback_cluster,
        int(raw.depth) if raw.depth is not None else int(fallback_depth_key),
    )


def _query_marginal_gain(
    gid: int,
    support_by_landmark: Mapping[int, Mapping[str, float]],
    query_coverage: Mapping[str, float],
    *,
    target_query_support: float,
    easy_query_gain_decay: float,
) -> tuple[float, dict[str, float]]:
    support = support_by_landmark.get(int(gid), {})
    if not support:
        return 0.0, {}
    target = max(1.0e-6, float(target_query_support))
    gain = 0.0
    used: dict[str, float] = {}
    for query_id, raw_value in support.items():
        value = max(0.0, float(raw_value))
        current = max(0.0, float(query_coverage.get(str(query_id), 0.0)))
        remaining = max(0.0, target - current)
        capped = min(value, remaining)
        excess = max(0.0, value - capped)
        effective = capped + max(0.0, float(easy_query_gain_decay)) * excess
        deficit_boost = 1.0 + (remaining / target)
        used[str(query_id)] = effective
        gain += effective * deficit_boost
    return float(gain), used


def _query_diversity_gain(
    gid: int,
    used_query_gain: Mapping[str, float],
    selected_query_cluster_counts: Mapping[str, Mapping[tuple[int, int, int], int]],
    selected_query_depth_counts: Mapping[str, Mapping[int, int]],
    observation_by_landmark: Mapping[int, Mapping[str, QueryObservation]],
    *,
    cluster: tuple[int, int, int],
    depth_key: int,
    query_coverage: Mapping[str, float],
    target_query_support: float,
) -> tuple[float, float]:
    target = max(1.0e-6, float(target_query_support))
    spatial_gain = 0.0
    depth_gain = 0.0
    for raw_query_id, raw_gain in used_query_gain.items():
        if max(0.0, float(raw_gain)) <= 0.0:
            continue
        query_id = str(raw_query_id)
        remaining = max(0.0, target - max(0.0, float(query_coverage.get(query_id, 0.0))))
        deficit_boost = 1.0 + (remaining / target)
        query_cluster, query_depth = _query_observation_key(
            int(gid),
            query_id,
            observation_by_landmark,
            fallback_cluster=cluster,
            fallback_depth_key=depth_key,
        )
        if selected_query_cluster_counts.get(query_id, {}).get(query_cluster, 0) == 0:
            spatial_gain += deficit_boost
        if selected_query_depth_counts.get(query_id, {}).get(int(query_depth), 0) == 0:
            depth_gain += deficit_boost
    return float(spatial_gain), float(depth_gain)


def _unit_vector(values: tuple[float, float, float]) -> torch.Tensor | None:
    vector = torch.tensor(values, dtype=torch.float32)
    norm = float(vector.norm().item())
    if not math.isfinite(norm) or norm <= 1.0e-8:
        return None
    return vector / norm


def _observation_reliability(observation: QueryObservation) -> float:
    reliability = 1.0
    if observation.reprojection_error_px is not None:
        reliability *= max(0.0, min(1.0, 1.0 - float(observation.reprojection_error_px) / 16.0))
    if observation.descriptor_margin is not None:
        reliability *= max(0.0, min(1.0, float(observation.descriptor_margin)))
    if observation.local_geometry_score is not None:
        reliability *= max(0.0, min(1.0, float(observation.local_geometry_score)))
    if observation.ray_artifact_score is not None:
        reliability *= max(0.0, min(1.0, 1.0 - float(observation.ray_artifact_score)))
    if observation.ray_entropy is not None:
        reliability *= max(0.0, min(1.0, 1.0 - 0.25 * float(observation.ray_entropy)))
    return float(max(0.0, min(1.0, reliability)))


def _query_pnp_gain(
    gid: int,
    used_query_gain: Mapping[str, float],
    selected_query_bearings: Mapping[str, list[torch.Tensor]],
    selected_query_camera_points: Mapping[str, list[torch.Tensor]],
    observation_by_landmark: Mapping[int, Mapping[str, QueryObservation]],
    *,
    distance_scale_m: float,
) -> tuple[float, float]:
    bearing_gain = 0.0
    geometry_gain = 0.0
    scale = max(1.0e-6, float(distance_scale_m))
    for raw_query_id, raw_gain in used_query_gain.items():
        if max(0.0, float(raw_gain)) <= 0.0:
            continue
        query_id = str(raw_query_id)
        observation = observation_by_landmark.get(int(gid), {}).get(query_id)
        if observation is None:
            continue
        reliability = _observation_reliability(observation)
        if reliability <= 0.0:
            continue

        if observation.bearing is not None:
            candidate_bearing = _unit_vector(observation.bearing)
            if candidate_bearing is not None:
                existing = selected_query_bearings.get(query_id, [])
                if not existing:
                    bearing_gain += reliability
                else:
                    best_cos = max(
                        -1.0,
                        min(1.0, max(float(torch.dot(candidate_bearing, other).item()) for other in existing)),
                    )
                    bearing_gain += reliability * math.sqrt(max(0.0, 1.0 - best_cos * best_cos))

        if observation.camera_xyz is not None:
            candidate_point = torch.tensor(observation.camera_xyz, dtype=torch.float32)
            if bool(torch.isfinite(candidate_point).all()):
                existing_points = selected_query_camera_points.get(query_id, [])
                if not existing_points:
                    geometry_gain += reliability
                else:
                    min_distance = min(float((candidate_point - point).norm().item()) for point in existing_points)
                    geometry_gain += reliability * min(1.0, max(0.0, min_distance / scale))
    return float(bearing_gain), float(geometry_gain)


def _max_pairwise_vector_spread(vectors: list[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    best = 0.0
    for i, first in enumerate(vectors[:-1]):
        for second in vectors[i + 1 :]:
            cosine = max(-1.0, min(1.0, float(torch.dot(first, second).item())))
            best = max(best, math.sqrt(max(0.0, 1.0 - cosine * cosine)))
    return float(best)


def _max_pairwise_distance(points: list[torch.Tensor]) -> float:
    if len(points) < 2:
        return 0.0
    best = 0.0
    for i, first in enumerate(points[:-1]):
        for second in points[i + 1 :]:
            best = max(best, float((first - second).norm().item()))
    return float(best)


def _query_validation_metrics(
    selected_ids: list[int] | set[int],
    query_ids: tuple[str, ...],
    support_by_landmark: Mapping[int, Mapping[str, float]],
    observation_by_landmark: Mapping[int, Mapping[str, QueryObservation]],
    xyz: torch.Tensor,
    *,
    spatial_cluster_size_m: float,
    depth_bin_size_m: float,
) -> dict[str, dict[str, float]]:
    selected_set = {int(gid) for gid in selected_ids}
    metrics: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        count = 0
        support_sum = 0.0
        cells: set[tuple[int, int, int]] = set()
        depths: set[int] = set()
        bearings: list[torch.Tensor] = []
        camera_points: list[torch.Tensor] = []
        reliability_sum = 0.0
        for gid in selected_set:
            value = support_by_landmark.get(int(gid), {}).get(str(query_id))
            if value is None or float(value) <= 0.0:
                continue
            count += 1
            support_sum += max(0.0, float(value))
            cluster = _cluster_key(xyz[int(gid)], float(spatial_cluster_size_m))
            depth_key = _depth_bin(xyz[int(gid)], float(depth_bin_size_m))
            query_cluster, query_depth = _query_observation_key(
                int(gid),
                str(query_id),
                observation_by_landmark,
                fallback_cluster=cluster,
                fallback_depth_key=depth_key,
            )
            cells.add(query_cluster)
            depths.add(int(query_depth))
            observation = observation_by_landmark.get(int(gid), {}).get(str(query_id))
            if observation is not None:
                reliability_sum += _observation_reliability(observation)
                if observation.bearing is not None:
                    bearing = _unit_vector(observation.bearing)
                    if bearing is not None:
                        bearings.append(bearing)
                if observation.camera_xyz is not None:
                    point = torch.tensor(observation.camera_xyz, dtype=torch.float32)
                    if bool(torch.isfinite(point).all()):
                        camera_points.append(point)
        metrics[str(query_id)] = {
            "landmark_count": float(count),
            "support_sum": float(support_sum),
            "cell_count": float(len(cells)),
            "depth_bin_count": float(len(depths)),
            "bearing_spread": _max_pairwise_vector_spread(bearings),
            "camera_spread_m": _max_pairwise_distance(camera_points),
            "mean_reliability": float(reliability_sum / count) if count > 0 else 0.0,
        }
    return metrics


def _query_validation_failures(
    selected_ids: list[int] | set[int],
    query_ids: tuple[str, ...],
    support_by_landmark: Mapping[int, Mapping[str, float]],
    observation_by_landmark: Mapping[int, Mapping[str, QueryObservation]],
    xyz: torch.Tensor,
    cfg: SparseSetSelectionConfig,
) -> tuple[set[str], dict[str, dict[str, float]]]:
    metrics = _query_validation_metrics(
        selected_ids,
        query_ids,
        support_by_landmark,
        observation_by_landmark,
        xyz,
        spatial_cluster_size_m=float(cfg.spatial_cluster_size_m),
        depth_bin_size_m=float(cfg.depth_bin_size_m),
    )
    failed: set[str] = set()
    min_landmarks = max(0, int(cfg.validation_min_query_landmarks))
    min_cells = max(0, int(cfg.validation_min_query_cells))
    min_depths = max(0, int(cfg.validation_min_query_depth_bins))
    min_bearing = max(0.0, float(cfg.validation_min_query_bearing_spread))
    min_camera = max(0.0, float(cfg.validation_min_query_camera_spread_m))
    if min_landmarks <= 0 and min_cells <= 0 and min_depths <= 0 and min_bearing <= 0.0 and min_camera <= 0.0:
        return failed, metrics
    for query_id, query_metrics in metrics.items():
        if min_landmarks > 0 and query_metrics["landmark_count"] < min_landmarks:
            failed.add(str(query_id))
        if min_cells > 0 and query_metrics["cell_count"] < min_cells:
            failed.add(str(query_id))
        if min_depths > 0 and query_metrics["depth_bin_count"] < min_depths:
            failed.add(str(query_id))
        if min_bearing > 0.0 and query_metrics["bearing_spread"] < min_bearing:
            failed.add(str(query_id))
        if min_camera > 0.0 and query_metrics["camera_spread_m"] < min_camera:
            failed.add(str(query_id))
    return failed, metrics


def _validation_gain_for_query(
    gid: int,
    query_id: str,
    selected_ids: list[int] | set[int],
    support_by_landmark: Mapping[int, Mapping[str, float]],
    observation_by_landmark: Mapping[int, Mapping[str, QueryObservation]],
    xyz: torch.Tensor,
    cfg: SparseSetSelectionConfig,
) -> float:
    before_failed, before_metrics = _query_validation_failures(
        selected_ids,
        (str(query_id),),
        support_by_landmark,
        observation_by_landmark,
        xyz,
        cfg,
    )
    after_failed, after_metrics = _query_validation_failures(
        set(int(v) for v in selected_ids) | {int(gid)},
        (str(query_id),),
        support_by_landmark,
        observation_by_landmark,
        xyz,
        cfg,
    )
    before = before_metrics.get(str(query_id), {})
    after = after_metrics.get(str(query_id), {})
    gain = 0.0
    for key in ("landmark_count", "cell_count", "depth_bin_count"):
        gain += max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0)))
    gain += 2.0 * max(0.0, float(after.get("bearing_spread", 0.0)) - float(before.get("bearing_spread", 0.0)))
    gain += max(0.0, float(after.get("camera_spread_m", 0.0)) - float(before.get("camera_spread_m", 0.0))) / max(
        1.0e-6, float(cfg.query_pnp_distance_scale_m)
    )
    if str(query_id) in before_failed and str(query_id) not in after_failed:
        gain += 10.0
    return float(gain)


def select_sparse_solver_set_from_full_gaussians(
    signals: SparseSetSignals,
    config: SparseSetSelectionConfig | None = None,
) -> SparseSetSelectionResult:
    """Greedy sparse-only full-Gaussian set selection.

    This selector is deliberately not a native/ULF replacement edit.  It builds a
    variable-size landmark set from all valid Gaussian ids and stops when
    marginal utility saturates or an optional maximum is reached.
    """

    cfg = config or SparseSetSelectionConfig()
    xyz = _xyz_tensor(signals.xyz)
    size = int(xyz.shape[0])
    kc = _float_vector(signals.kc_score, name="kc_score", size=size)
    visibility = _float_vector(signals.visibility_score, name="visibility_score", size=size)
    mask_validity = _float_vector(signals.mask_validity, name="mask_validity", size=size)
    solver = _float_vector(signals.solver_support, name="solver_support", size=size)
    risk = (
        torch.zeros(size, dtype=torch.float32)
        if signals.ambiguity_risk is None
        else _float_vector(signals.ambiguity_risk, name="ambiguity_risk", size=size)
    )
    sparse_validation_risk = (
        torch.zeros(size, dtype=torch.float32)
        if signals.sparse_validation_risk is None
        else _float_vector(signals.sparse_validation_risk, name="sparse_validation_risk", size=size)
    )
    sparse_validation_hard_reject_exempt = (
        None
        if signals.sparse_validation_hard_reject_exempt is None
        else _bool_vector(
            signals.sparse_validation_hard_reject_exempt,
            name="sparse_validation_hard_reject_exempt",
            size=size,
        )
    )
    support_match_strength = (
        torch.zeros(size, dtype=torch.float32)
        if signals.support_match_strength is None
        else _float_vector(signals.support_match_strength, name="support_match_strength", size=size)
    )
    match_competition_risk = (
        torch.zeros(size, dtype=torch.float32)
        if signals.match_competition_risk is None
        else _float_vector(signals.match_competition_risk, name="match_competition_risk", size=size)
    )
    descriptor_features = _feature_matrix(signals.descriptor_features, name="descriptor_features", size=size)
    source_anchor_idx = (
        torch.empty(0, dtype=torch.long)
        if signals.source_anchor_idx is None
        else torch.as_tensor(signals.source_anchor_idx, dtype=torch.long).reshape(-1).cpu()
    )
    source_anchor_set = {int(gid) for gid in source_anchor_idx.tolist() if 0 <= int(gid) < size}
    source_anchor_mask = torch.zeros(size, dtype=torch.bool)
    for gid in source_anchor_set:
        source_anchor_mask[int(gid)] = True
    source_anchor_mode = str(cfg.source_anchor_mode).strip().lower()
    if source_anchor_mode not in {"hard", "soft", "off"}:
        raise ValueError(f"source_anchor_mode must be one of hard/soft/off, got {cfg.source_anchor_mode!r}")
    source_anchor_soft_prior_scale = max(1.0, float(cfg.target_query_support))
    source_anchor_prior = torch.zeros(size, dtype=torch.float32)
    for gid in source_anchor_set:
        source_anchor_prior[int(gid)] = 1.0

    sparse_validation_risk_original_nonzero_count = int((sparse_validation_risk > 0).sum().item())
    sparse_validation_risk_source_anchor_exempt = bool(cfg.sparse_validation_risk_source_anchor_exempt)
    sparse_validation_risk_source_anchor_exempt_count = 0
    if sparse_validation_risk_source_anchor_exempt:
        if not source_anchor_set:
            raise ValueError("source_anchor_idx is required when sparse_validation_risk_source_anchor_exempt=True")
        source_risk_mask = source_anchor_mask & (sparse_validation_risk > 0.0)
        sparse_validation_risk_source_anchor_exempt_count = int(source_risk_mask.sum().item())
        sparse_validation_risk = torch.where(
            source_anchor_mask,
            torch.zeros_like(sparse_validation_risk),
            sparse_validation_risk,
        )

    finite = (
        torch.isfinite(xyz).all(dim=1)
        & torch.isfinite(kc)
        & torch.isfinite(visibility)
        & torch.isfinite(mask_validity)
        & torch.isfinite(solver)
        & torch.isfinite(risk)
        & torch.isfinite(sparse_validation_risk)
        & torch.isfinite(support_match_strength)
        & torch.isfinite(match_competition_risk)
    )
    mask_ok = mask_validity >= float(cfg.min_mask_validity)
    visibility_ok = visibility >= float(cfg.min_visibility)
    support_by_landmark, query_ids = _canonical_query_support(signals.per_query_support, size=size)
    validation_support_by_landmark, validation_query_ids = _canonical_query_support(
        signals.validation_per_query_support,
        size=size,
    )
    validation_protected_mask = torch.zeros(size, dtype=torch.bool)
    for gid, query_map in validation_support_by_landmark.items():
        if any(float(value) > 0.0 for value in query_map.values()):
            validation_protected_mask[int(gid)] = True
    sparse_validation_prune_protected_count = int(validation_protected_mask.sum().item())

    def is_sparse_validation_protected(gid: int) -> bool:
        return 0 <= int(gid) < size and bool(validation_protected_mask[int(gid)].item())

    hard_reject_exempt_mask = (
        validation_protected_mask
        if sparse_validation_hard_reject_exempt is None
        else sparse_validation_hard_reject_exempt
    )
    sparse_validation_hard_reject_exempt_count = int(hard_reject_exempt_mask.sum().item())
    sparse_validation_risk_protected_hard_reject_exempt = bool(
        cfg.sparse_validation_risk_protected_hard_reject_exempt
    )
    sparse_validation_risk_threshold = max(0.0, float(cfg.sparse_validation_risk_reject_threshold))
    sparse_validation_risk_ok = (
        torch.ones(size, dtype=torch.bool)
        if sparse_validation_risk_threshold <= 0.0
        else (sparse_validation_risk <= sparse_validation_risk_threshold)
        | (hard_reject_exempt_mask if sparse_validation_risk_protected_hard_reject_exempt else False)
    )
    sparse_validation_risk_hard_reject_exempt_count = int(
        (
            finite
            & mask_ok
            & visibility_ok
            & hard_reject_exempt_mask
            & (sparse_validation_risk > sparse_validation_risk_threshold)
        ).sum().item()
    ) if sparse_validation_risk_threshold > 0.0 and sparse_validation_risk_protected_hard_reject_exempt else 0
    match_competition_risk_threshold = max(0.0, float(cfg.match_competition_risk_reject_threshold))
    match_competition_risk_ok = (
        torch.ones(size, dtype=torch.bool)
        if match_competition_risk_threshold <= 0.0
        else match_competition_risk <= match_competition_risk_threshold
    )
    valid = finite & mask_ok & visibility_ok & sparse_validation_risk_ok & match_competition_risk_ok

    if validation_support_by_landmark:
        merged_support_by_landmark = {int(gid): dict(query_map) for gid, query_map in support_by_landmark.items()}
        for gid, query_map in validation_support_by_landmark.items():
            merged_query_map = merged_support_by_landmark.setdefault(int(gid), {})
            for query_id, value in query_map.items():
                merged_query_map[str(query_id)] = float(merged_query_map.get(str(query_id), 0.0) + float(value))
        support_by_landmark = merged_support_by_landmark
    if validation_query_ids:
        query_ids = tuple(sorted(set(query_ids) | set(validation_query_ids)))
    query_match_strength_by_landmark, query_match_strength_query_ids = _canonical_query_support(
        signals.per_query_match_strength,
        size=size,
    )
    query_match_risk_by_landmark, query_match_risk_query_ids = _canonical_query_support(
        signals.per_query_match_competition_risk,
        size=size,
    )
    query_match_risk_reference = _canonical_query_scalar_table(
        signals.per_query_match_competition_risk_reference
    )
    query_match_competition_reference_landmarks = _canonical_query_landmark_sets(
        signals.per_query_match_competition_reference_landmarks,
        size=size,
    )
    if (
        query_match_strength_query_ids
        or query_match_risk_query_ids
        or query_match_risk_reference
        or query_match_competition_reference_landmarks
    ):
        query_ids = tuple(
            sorted(
                set(query_ids)
                | set(query_match_strength_query_ids)
                | set(query_match_risk_query_ids)
                | set(query_match_risk_reference.keys())
                | set(query_match_competition_reference_landmarks.keys())
            )
        )
    query_match_strength_by_query = _query_table_by_query(query_match_strength_by_landmark)
    query_match_risk_by_query = _query_table_by_query(query_match_risk_by_landmark)
    query_match_strength_total_by_query = {
        str(query_id): float(sum(value for _gid, value in items))
        for query_id, items in query_match_strength_by_query.items()
    }
    observation_by_landmark = _canonical_query_observations(signals.per_query_observations, size=size)
    explicit_descriptor_conflict_edges = _canonical_descriptor_conflict_edges(
        signals.descriptor_conflict_edges,
        size=size,
    )
    descriptor_conflict_edges = explicit_descriptor_conflict_edges
    auto_descriptor_conflict_edge_count = 0
    query_raw_support = torch.zeros(size, dtype=torch.float32)
    for gid, query_map in support_by_landmark.items():
        query_raw_support[int(gid)] = float(sum(max(0.0, float(v)) for v in query_map.values()))

    def compute_static_score(active_sparse_validation_risk: torch.Tensor, active_valid: torch.Tensor) -> torch.Tensor:
        score = (
            float(cfg.kc_weight) * _normalize(kc)
            + float(cfg.visibility_weight) * _normalize(visibility)
            + float(cfg.mask_weight) * _normalize(mask_validity)
            + float(cfg.solver_weight) * _normalize(solver)
            + float(cfg.query_support_weight) * _normalize(query_raw_support)
            + float(cfg.support_match_strength_weight) * _normalize(support_match_strength)
            + float(cfg.source_anchor_weight) * source_anchor_soft_prior_scale * source_anchor_prior
            - float(cfg.ambiguity_risk_weight) * _normalize(risk)
            - float(cfg.sparse_validation_risk_weight) * _normalize(active_sparse_validation_risk)
            - float(cfg.match_competition_risk_weight) * _normalize(match_competition_risk)
        )
        return torch.where(active_valid, score, torch.full_like(score, -float("inf")))

    static_score = compute_static_score(sparse_validation_risk, valid)

    valid_candidate_ids = torch.where(valid)[0]
    if valid_candidate_ids.numel() == 0:
        return SparseSetSelectionResult(
            sampled_idx=torch.empty(0, dtype=torch.long),
            metadata={
                "selection_policy": "full_gaussian_sparse_set",
                "source": "full_gaussians",
                "budget_mode": "variable_size_marginal_gain",
                "input_gaussian_count": size,
                "selected_count": 0,
                "candidate_count": 0,
                "invalid_mask_filtered_count": int((finite & ~mask_ok).sum().item()),
                "low_visibility_filtered_count": int((finite & mask_ok & ~visibility_ok).sum().item()),
                "high_match_competition_risk_filtered_count": int(
                    (finite & mask_ok & visibility_ok & sparse_validation_risk_ok & ~match_competition_risk_ok)
                    .sum()
                    .item()
                ),
            },
        )

    pool_size = cfg.candidate_pool_size
    if pool_size is not None and int(pool_size) > 0 and int(valid_candidate_ids.numel()) > int(pool_size):
        _, top_local = torch.topk(static_score[valid_candidate_ids], k=int(pool_size), largest=True)
        candidate_ids = valid_candidate_ids[top_local]
    else:
        candidate_ids = valid_candidate_ids
    candidate_id_set = {int(gid) for gid in candidate_ids.tolist()}
    candidate_pool_static_count = int(len(candidate_id_set))
    candidate_pool_source_anchor_added_count = 0
    candidate_pool_query_added_count = 0
    candidate_pool_kc_added_count = 0
    candidate_pool_visibility_added_count = 0
    candidate_pool_mask_added_count = 0
    candidate_pool_solver_added_count = 0
    candidate_pool_match_strength_added_count = 0
    candidate_pool_query_match_strength_added_count = 0

    def add_candidate(gid: int, *, reason: str = "query") -> None:
        nonlocal candidate_pool_source_anchor_added_count, candidate_pool_query_added_count
        nonlocal candidate_pool_kc_added_count, candidate_pool_visibility_added_count
        nonlocal candidate_pool_mask_added_count, candidate_pool_solver_added_count
        nonlocal candidate_pool_match_strength_added_count
        nonlocal candidate_pool_query_match_strength_added_count
        gid = int(gid)
        if gid < 0 or gid >= size or not bool(valid[gid].item()) or gid in candidate_id_set:
            return
        candidate_id_set.add(gid)
        if reason == "source_anchor":
            candidate_pool_source_anchor_added_count += 1
        elif reason == "kc":
            candidate_pool_kc_added_count += 1
        elif reason == "visibility":
            candidate_pool_visibility_added_count += 1
        elif reason == "mask":
            candidate_pool_mask_added_count += 1
        elif reason == "solver":
            candidate_pool_solver_added_count += 1
        elif reason == "match_strength":
            candidate_pool_match_strength_added_count += 1
        elif reason == "query_match_strength":
            candidate_pool_query_match_strength_added_count += 1
        else:
            candidate_pool_query_added_count += 1

    for gid in source_anchor_set:
        add_candidate(int(gid), reason="source_anchor")

    def add_component_topk(values: torch.Tensor, *, top_k: int, reason: str) -> None:
        k = max(0, int(top_k))
        if k <= 0:
            return
        component_score = torch.where(valid, values.float(), torch.full_like(values.float(), -float("inf")))
        finite_component_ids = torch.where(torch.isfinite(component_score))[0]
        if finite_component_ids.numel() == 0:
            return
        k = min(k, int(finite_component_ids.numel()))
        _, top_local = torch.topk(component_score[finite_component_ids], k=k, largest=True)
        for gid in finite_component_ids[top_local].tolist():
            add_candidate(int(gid), reason=reason)

    candidate_pool_kc_top_k = max(0, int(cfg.candidate_pool_kc_top_k))
    candidate_pool_visibility_top_k = max(0, int(cfg.candidate_pool_visibility_top_k))
    candidate_pool_mask_top_k = max(0, int(cfg.candidate_pool_mask_top_k))
    candidate_pool_solver_top_k = max(0, int(cfg.candidate_pool_solver_top_k))
    candidate_pool_match_strength_top_k = max(0, int(cfg.candidate_pool_match_strength_top_k))
    add_component_topk(kc, top_k=candidate_pool_kc_top_k, reason="kc")
    add_component_topk(visibility, top_k=candidate_pool_visibility_top_k, reason="visibility")
    add_component_topk(mask_validity, top_k=candidate_pool_mask_top_k, reason="mask")
    add_component_topk(solver, top_k=candidate_pool_solver_top_k, reason="solver")
    add_component_topk(
        support_match_strength,
        top_k=candidate_pool_match_strength_top_k,
        reason="match_strength",
    )

    query_match_strength_top_k = max(0, int(cfg.candidate_pool_query_match_strength_top_k))
    if query_match_strength_top_k > 0:
        for query_id, items in query_match_strength_by_query.items():
            ranked = [
                (int(gid), float(value))
                for gid, value in items
                if 0 <= int(gid) < size and bool(valid[int(gid)].item())
            ]
            ranked.sort(
                key=lambda item: (
                    float(item[1]),
                    float(static_score[int(item[0])].item()),
                ),
                reverse=True,
            )
            for gid, _value in ranked[:query_match_strength_top_k]:
                add_candidate(int(gid), reason="query_match_strength")

    query_top_k = max(0, int(cfg.candidate_pool_query_top_k))
    query_cell_top_k = max(0, int(cfg.candidate_pool_query_cell_top_k))
    query_depth_top_k = max(0, int(cfg.candidate_pool_query_depth_top_k))
    if query_top_k > 0 or query_cell_top_k > 0 or query_depth_top_k > 0:
        for query_id in query_ids:
            query_items: list[tuple[int, float]] = []
            for gid, query_map in support_by_landmark.items():
                if query_id in query_map and bool(valid[int(gid)].item()):
                    query_items.append((int(gid), float(query_map[query_id])))
            query_items.sort(key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())), reverse=True)
            if query_top_k > 0:
                for gid, _value in query_items[:query_top_k]:
                    add_candidate(gid, reason="query")
            if query_cell_top_k > 0 or query_depth_top_k > 0:
                by_cell: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
                by_depth: dict[int, list[tuple[int, float]]] = {}
                for gid, value in query_items:
                    cluster = _cluster_key(xyz[int(gid)], float(cfg.spatial_cluster_size_m))
                    depth_key = _depth_bin(xyz[int(gid)], float(cfg.depth_bin_size_m))
                    query_cluster, query_depth = _query_observation_key(
                        int(gid),
                        query_id,
                        observation_by_landmark,
                        fallback_cluster=cluster,
                        fallback_depth_key=depth_key,
                    )
                    by_cell.setdefault(query_cluster, []).append((int(gid), float(value)))
                    by_depth.setdefault(int(query_depth), []).append((int(gid), float(value)))
                if query_cell_top_k > 0:
                    for items in by_cell.values():
                        items.sort(key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())), reverse=True)
                        for gid, _value in items[:query_cell_top_k]:
                            add_candidate(gid, reason="query")
                if query_depth_top_k > 0:
                    for items in by_depth.values():
                        items.sort(key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())), reverse=True)
                        for gid, _value in items[:query_depth_top_k]:
                            add_candidate(gid, reason="query")

    candidate_ids = torch.tensor(sorted(candidate_id_set), dtype=torch.long)
    sparse_validation_risk_expansion_metadata: dict[str, float | int] = {
        "sparse_validation_risk_expansion_seed_count": 0,
        "sparse_validation_risk_expansion_candidate_count": 0,
        "sparse_validation_risk_expanded_count": 0,
        "sparse_validation_risk_expansion_edge_count": 0,
        "sparse_validation_risk_expansion_max": float(sparse_validation_risk.max().item())
        if int(sparse_validation_risk.numel()) > 0
        else 0.0,
    }
    if int(cfg.sparse_validation_risk_expand_top_k) > 0 and int(candidate_ids.numel()) > 0:
        pre_expansion_order = torch.argsort(static_score[candidate_ids], descending=True)
        expansion_candidate_ids = candidate_ids[pre_expansion_order]
        sparse_validation_risk, sparse_validation_risk_expansion_metadata = (
            _expand_sparse_validation_risk_by_descriptor(
                descriptor_features,
                xyz,
                sparse_validation_risk,
                expansion_candidate_ids,
                top_k=int(cfg.sparse_validation_risk_expand_top_k),
                neighbors_per_seed=int(cfg.sparse_validation_risk_expand_neighbors_per_seed),
                candidate_limit=int(cfg.sparse_validation_risk_expand_candidate_limit),
                min_cosine=float(cfg.sparse_validation_risk_expand_min_cosine),
                min_spatial_distance_m=float(cfg.sparse_validation_risk_expand_min_spatial_distance_m),
                weight=float(cfg.sparse_validation_risk_expand_weight),
            )
        )
        sparse_validation_risk_ok = (
            torch.ones(size, dtype=torch.bool)
            if sparse_validation_risk_threshold <= 0.0
            else (sparse_validation_risk <= sparse_validation_risk_threshold)
            | (hard_reject_exempt_mask if sparse_validation_risk_protected_hard_reject_exempt else False)
        )
        sparse_validation_risk_hard_reject_exempt_count = int(
            (
                finite
                & mask_ok
                & visibility_ok
                & hard_reject_exempt_mask
                & (sparse_validation_risk > sparse_validation_risk_threshold)
            ).sum().item()
        ) if sparse_validation_risk_threshold > 0.0 and sparse_validation_risk_protected_hard_reject_exempt else 0
        valid = finite & mask_ok & visibility_ok & sparse_validation_risk_ok & match_competition_risk_ok
        static_score = compute_static_score(sparse_validation_risk, valid)
        candidate_id_set = {int(gid) for gid in candidate_id_set if bool(valid[int(gid)].item())}
        candidate_ids = torch.tensor(sorted(candidate_id_set), dtype=torch.long)
    order = torch.argsort(static_score[candidate_ids], descending=True)
    ordered_ids = [int(candidate_ids[int(i)].item()) for i in order.tolist()]
    auto_descriptor_conflict_edges = _build_auto_descriptor_conflict_edges(
        descriptor_features,
        xyz,
        torch.tensor(ordered_ids, dtype=torch.long),
        top_k=int(cfg.auto_descriptor_conflict_top_k),
        candidate_limit=int(cfg.auto_descriptor_conflict_candidate_limit),
        min_cosine=float(cfg.auto_descriptor_conflict_min_cosine),
        min_spatial_distance_m=float(cfg.auto_descriptor_conflict_min_spatial_distance_m),
    )
    auto_descriptor_conflict_edge_count = sum(len(neighbors) for neighbors in auto_descriptor_conflict_edges.values())
    descriptor_conflict_edges = _merge_descriptor_conflict_edges(
        explicit_descriptor_conflict_edges,
        auto_descriptor_conflict_edges,
        size=size,
    )
    descriptor_conflict_edge_count = sum(len(neighbors) for neighbors in descriptor_conflict_edges.values())

    selected: list[int] = []
    selected_set: set[int] = set()
    cluster_counts: dict[tuple[int, int, int], int] = {}
    used_clusters: set[tuple[int, int, int]] = set()
    used_depth_bins: set[int] = set()
    query_coverage = {query_id: 0.0 for query_id in query_ids}
    selected_query_counts = {query_id: 0 for query_id in query_ids}
    selected_query_cluster_counts: dict[str, dict[tuple[int, int, int], int]] = {
        query_id: {} for query_id in query_ids
    }
    selected_query_depth_counts: dict[str, dict[int, int]] = {query_id: {} for query_id in query_ids}
    selected_query_bearings: dict[str, list[torch.Tensor]] = {query_id: [] for query_id in query_ids}
    selected_query_camera_points: dict[str, list[torch.Tensor]] = {query_id: [] for query_id in query_ids}
    selected_query_match_strength = {query_id: 0.0 for query_id in query_ids}
    selected_query_match_risk = {query_id: 0.0 for query_id in query_ids}
    marginal_scores: list[float] = []
    cluster_cap_rejected = 0
    low_gain_rejected = 0
    kc_anchor_selected = 0
    query_prefill_selected = 0
    source_query_prefill_selected = 0
    validation_query_prefill_selected = 0
    source_anchor_selected = 0
    augmentation_solver_gate_rejected = 0
    augmentation_query_gate_rejected = 0
    augmentation_kc_gate_rejected = 0
    main_query_gain_gate_rejected = 0
    main_query_match_context_gate_accepted = 0

    max_landmarks = None if cfg.max_landmarks is None or int(cfg.max_landmarks) <= 0 else int(cfg.max_landmarks)
    min_landmarks = max(0, int(cfg.min_landmarks))
    local_search_rounds = max(0, int(cfg.local_search_rounds))
    local_search_candidates_per_query = max(0, int(cfg.local_search_candidates_per_query))
    local_search_max_additions = max(0, int(cfg.local_search_max_additions))
    local_search_reserve_count = max(0, int(cfg.local_search_reserve_count))
    pre_local_max_landmarks = max_landmarks
    if (
        max_landmarks is not None
        and local_search_rounds > 0
        and local_search_candidates_per_query > 0
        and local_search_reserve_count > 0
    ):
        pre_local_max_landmarks = max(
            min_landmarks,
            int(max_landmarks) - min(int(local_search_reserve_count), int(max_landmarks)),
        )
    max_per_cluster = max(0, int(cfg.max_per_spatial_cluster))
    force_source_anchor = bool(cfg.force_source_anchor)
    min_aug_solver = max(0.0, float(cfg.augmentation_min_solver_support))
    min_aug_query = max(0.0, float(cfg.augmentation_min_query_support))
    min_aug_kc = max(0.0, float(cfg.augmentation_min_kc_score))
    query_prefill_hard_coverage_bonus = max(0.0, float(cfg.query_prefill_hard_coverage_bonus))
    def can_select(gid: int, *, source_anchor: bool = False) -> tuple[bool, tuple[int, int, int], int, str]:
        if gid in selected_set:
            return False, (0, 0, 0), 0, "selected"
        valid_mask = finite[int(gid)] if source_anchor and force_source_anchor else valid[int(gid)]
        if not bool(valid_mask.item()):
            return False, (0, 0, 0), 0, "invalid"
        point = xyz[int(gid)]
        cluster = _cluster_key(point, float(cfg.spatial_cluster_size_m))
        if not (source_anchor and force_source_anchor) and max_per_cluster > 0 and cluster_counts.get(cluster, 0) >= max_per_cluster:
            return False, cluster, 0, "cluster_cap"
        return True, cluster, _depth_bin(point, float(cfg.depth_bin_size_m)), ""

    def passes_augmentation_gate(gid: int) -> tuple[bool, str]:
        if int(gid) in source_anchor_set:
            return True, ""
        if min_aug_kc > 0.0 and float(kc[int(gid)].item()) < min_aug_kc:
            return False, "kc"
        if min_aug_solver > 0.0 and float(solver[int(gid)].item()) < min_aug_solver:
            return False, "solver"
        if min_aug_query > 0.0 and float(query_raw_support[int(gid)].item()) < min_aug_query:
            return False, "query"
        return True, ""

    def accept(gid: int, *, marginal: float, used_query_gain: Mapping[str, float], cluster: tuple[int, int, int], depth_key: int) -> None:
        selected.append(int(gid))
        selected_set.add(int(gid))
        marginal_scores.append(float(marginal))
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        used_clusters.add(cluster)
        used_depth_bins.add(depth_key)
        for query_id, value in used_query_gain.items():
            query_key = str(query_id)
            query_cluster, query_depth = _query_observation_key(
                int(gid),
                query_key,
                observation_by_landmark,
                fallback_cluster=cluster,
                fallback_depth_key=depth_key,
            )
            query_coverage[query_key] = float(query_coverage.get(query_key, 0.0)) + max(0.0, float(value))
            selected_query_counts[query_key] = int(selected_query_counts.get(query_key, 0)) + 1
            per_query_clusters = selected_query_cluster_counts.setdefault(query_key, {})
            per_query_clusters[query_cluster] = per_query_clusters.get(query_cluster, 0) + 1
            per_query_depths = selected_query_depth_counts.setdefault(query_key, {})
            per_query_depths[int(query_depth)] = per_query_depths.get(int(query_depth), 0) + 1
            observation = observation_by_landmark.get(int(gid), {}).get(query_key)
            if observation is not None and observation.bearing is not None:
                bearing = _unit_vector(observation.bearing)
                if bearing is not None:
                    selected_query_bearings.setdefault(query_key, []).append(bearing)
            if observation is not None and observation.camera_xyz is not None:
                point = torch.tensor(observation.camera_xyz, dtype=torch.float32)
                if bool(torch.isfinite(point).all()):
                    selected_query_camera_points.setdefault(query_key, []).append(point)
        for query_id, value in query_match_strength_by_landmark.get(int(gid), {}).items():
            query_key = str(query_id)
            selected_query_match_strength[query_key] = float(
                selected_query_match_strength.get(query_key, 0.0) + max(0.0, float(value))
            )
        for query_id, value in query_match_risk_by_landmark.get(int(gid), {}).items():
            query_key = str(query_id)
            selected_query_match_risk[query_key] = float(
                selected_query_match_risk.get(query_key, 0.0) + max(0.0, float(value))
            )

    def query_match_dominance_delta(gid: int) -> tuple[float, float, float, float, float]:
        """Return query-level match gain, risk penalty, and risk-ratio penalty.

        Static stealer vectors are too blunt: a Gaussian is harmful only in the
        query contexts where it competes against solver-supported landmarks.
        This marginal compares the candidate against the current selected set
        for those same queries.
        """

        strength_map = query_match_strength_by_landmark.get(int(gid), {})
        risk_map = query_match_risk_by_landmark.get(int(gid), {})
        if not strength_map and not risk_map:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        target = max(1.0e-6, float(cfg.target_query_support))
        strength_gain = 0.0
        for raw_query_id, raw_value in strength_map.items():
            query_id = str(raw_query_id)
            value = max(0.0, float(raw_value))
            if value <= 0.0:
                continue
            current = max(0.0, float(selected_query_match_strength.get(query_id, 0.0)))
            remaining = max(0.0, target - current)
            capped = min(value, remaining)
            excess = max(0.0, value - capped)
            effective = capped + max(0.0, float(cfg.easy_query_gain_decay)) * excess
            strength_gain += effective * (1.0 + remaining / target)

        risk_penalty = 0.0
        ratio_penalty = 0.0
        reference_penalty = 0.0
        nonreference_penalty = 0.0
        reference_margin = max(0.0, float(cfg.query_match_risk_reference_margin))
        for raw_query_id, raw_value in risk_map.items():
            query_id = str(raw_query_id)
            value = max(0.0, float(raw_value))
            if value <= 0.0:
                continue
            candidate_strength = max(0.0, float(strength_map.get(query_id, 0.0)))
            current_strength = max(0.0, float(selected_query_match_strength.get(query_id, 0.0)))
            current_risk = max(0.0, float(selected_query_match_risk.get(query_id, 0.0)))
            after_strength = current_strength + candidate_strength
            after_risk = current_risk + value
            available_strength = max(0.0, float(query_match_strength_total_by_query.get(query_id, 0.0)))
            support_context = min(1.0, max(current_strength, candidate_strength, available_strength) / target)
            risk_penalty += value * (0.25 + support_context)

            before_ratio = current_risk / max(1.0, current_strength)
            after_ratio = after_risk / max(1.0, after_strength)
            ratio_scale = min(target, max(1.0, current_strength, candidate_strength, available_strength))
            ratio_penalty += max(0.0, after_ratio - before_ratio) * ratio_scale
            if query_id in query_match_risk_reference:
                cap = max(0.0, float(query_match_risk_reference[query_id])) + reference_margin
                before_over = max(0.0, current_risk - cap)
                after_over = max(0.0, after_risk - cap)
                reference_penalty += max(0.0, after_over - before_over)
            reference_landmarks = query_match_competition_reference_landmarks.get(query_id)
            if reference_landmarks is not None and int(gid) not in reference_landmarks:
                nonreference_penalty += value
        return (
            float(strength_gain),
            float(risk_penalty),
            float(ratio_penalty),
            float(reference_penalty),
            float(nonreference_penalty),
        )

    def pnp_aware_marginal(
        gid: int,
        *,
        used_query_gain: Mapping[str, float],
        cluster: tuple[int, int, int],
        depth_key: int,
        spatial_gain: float,
        depth_gain: float,
        pnp_bootstrap_gain: float = 0.0,
    ) -> float:
        query_spatial_gain, query_depth_gain = _query_diversity_gain(
            int(gid),
            used_query_gain,
            selected_query_cluster_counts,
            selected_query_depth_counts,
            observation_by_landmark,
            cluster=cluster,
            depth_key=depth_key,
            query_coverage=query_coverage,
            target_query_support=float(cfg.target_query_support),
        )
        query_bearing_gain, query_pnp_geometry_gain = _query_pnp_gain(
            int(gid),
            used_query_gain,
            selected_query_bearings,
            selected_query_camera_points,
            observation_by_landmark,
            distance_scale_m=float(cfg.query_pnp_distance_scale_m),
        )
        (
            match_strength_gain,
            match_risk_penalty,
            match_ratio_penalty,
            match_reference_penalty,
            match_nonreference_penalty,
        ) = query_match_dominance_delta(int(gid))
        conflict_penalty = _descriptor_conflict_penalty(int(gid), selected_set, descriptor_conflict_edges)
        return float(
            float(static_score[int(gid)].item())
            + float(cfg.geometry_weight) * (float(spatial_gain) + float(pnp_bootstrap_gain))
            + float(cfg.depth_weight) * float(depth_gain)
            + float(cfg.query_geometry_weight) * query_spatial_gain
            + float(cfg.query_depth_weight) * query_depth_gain
            + float(cfg.query_bearing_weight) * query_bearing_gain
            + float(cfg.query_pnp_geometry_weight) * query_pnp_geometry_gain
            + float(cfg.query_match_strength_weight) * match_strength_gain
            - float(cfg.query_match_competition_weight) * match_risk_penalty
            - float(cfg.query_match_competition_ratio_weight) * match_ratio_penalty
            - float(cfg.query_match_risk_reference_weight) * match_reference_penalty
            - float(cfg.query_match_nonreference_competition_weight) * match_nonreference_penalty
            - float(cfg.descriptor_conflict_weight) * conflict_penalty
        )

    def soft_source_anchor_gate_gain(gid: int) -> float:
        if source_anchor_mode != "soft" or int(gid) not in source_anchor_set:
            return 0.0
        return float(cfg.source_anchor_weight) * float(source_anchor_soft_prior_scale)

    source_anchor_count = int(source_anchor_idx.numel())
    if source_anchor_count > 0 and source_anchor_mode == "hard":
        seen_anchor: set[int] = set()
        for raw_gid in source_anchor_idx.tolist():
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            gid = int(raw_gid)
            if gid in seen_anchor:
                continue
            seen_anchor.add(gid)
            ok, cluster, depth_key, _reason = can_select(gid, source_anchor=True)
            if not ok:
                continue
            query_gain, used_query_gain = _query_marginal_gain(
                gid,
                support_by_landmark,
                query_coverage,
                target_query_support=float(cfg.target_query_support),
                easy_query_gain_decay=float(cfg.easy_query_gain_decay),
            )
            spatial_gain = 1.0 if cluster not in used_clusters else 0.0
            depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
            marginal = pnp_aware_marginal(
                gid,
                used_query_gain=used_query_gain,
                cluster=cluster,
                depth_key=depth_key,
                spatial_gain=spatial_gain,
                depth_gain=depth_gain,
            ) + float(cfg.query_support_weight) * query_gain
            accept(gid, marginal=float(marginal), used_query_gain=used_query_gain, cluster=cluster, depth_key=depth_key)
            source_anchor_selected += 1

    source_query_prefill_fraction = max(0.0, float(cfg.source_query_prefill_target_fraction))
    source_query_prefill_max_candidates = max(0, int(cfg.source_query_prefill_max_candidates_per_query))
    source_query_prefill_max_per_cluster = max(0, int(cfg.source_query_prefill_max_per_spatial_cluster))
    source_query_prefill_min_cells = max(0, int(cfg.source_query_prefill_min_cells_per_query))
    source_query_prefill_min_depth_bins = max(0, int(cfg.source_query_prefill_min_depth_bins_per_query))

    def source_query_prefill_done(query_id: str, query_target: float) -> bool:
        coverage_done = float(query_coverage.get(str(query_id), 0.0)) >= float(query_target)
        cell_done = (
            source_query_prefill_min_cells <= 0
            or len(selected_query_cluster_counts.get(str(query_id), {})) >= source_query_prefill_min_cells
        )
        depth_done = (
            source_query_prefill_min_depth_bins <= 0
            or len(selected_query_depth_counts.get(str(query_id), {})) >= source_query_prefill_min_depth_bins
        )
        return bool(coverage_done and cell_done and depth_done)

    if source_query_prefill_fraction > 0.0 and source_anchor_set and support_by_landmark:
        by_query_source: dict[str, list[tuple[int, float]]] = {query_id: [] for query_id in query_ids}
        for gid, query_map in support_by_landmark.items():
            if int(gid) not in source_anchor_set or not bool(valid[int(gid)].item()):
                continue
            for query_id, value in query_map.items():
                raw_value = max(0.0, float(value))
                if raw_value > 0.0:
                    by_query_source.setdefault(str(query_id), []).append((int(gid), raw_value))
        for query_id in sorted(by_query_source):
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            candidates = by_query_source.get(str(query_id), [])
            if not candidates:
                continue
            total_support = sum(value for _, value in candidates)
            query_target = source_query_prefill_fraction * min(float(cfg.target_query_support), float(total_support))
            if query_target <= 0.0:
                continue
            ranked = sorted(
                candidates,
                key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())),
                reverse=True,
            )
            if source_query_prefill_max_candidates > 0:
                ranked = ranked[:source_query_prefill_max_candidates]
            pool = [int(gid) for gid, _value in ranked]
            while pool:
                if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                    break
                if source_query_prefill_done(str(query_id), float(query_target)):
                    break
                best: tuple[float, int, dict[str, float], tuple[int, int, int], int] | None = None
                next_pool: list[int] = []
                for gid in pool:
                    ok, cluster, depth_key, _reason = can_select(int(gid), source_anchor=True)
                    if not ok:
                        continue
                    query_cluster, _query_depth = _query_observation_key(
                        int(gid),
                        str(query_id),
                        observation_by_landmark,
                        fallback_cluster=cluster,
                        fallback_depth_key=depth_key,
                    )
                    query_cluster_counts = selected_query_cluster_counts.get(str(query_id), {})
                    if (
                        source_query_prefill_max_per_cluster > 0
                        and query_cluster_counts.get(query_cluster, 0) >= source_query_prefill_max_per_cluster
                    ):
                        continue
                    query_gain, used_query_gain = _query_marginal_gain(
                        int(gid),
                        support_by_landmark,
                        query_coverage,
                        target_query_support=float(cfg.target_query_support),
                        easy_query_gain_decay=float(cfg.easy_query_gain_decay),
                    )
                    if query_gain <= 0.0 and str(query_id) not in support_by_landmark.get(int(gid), {}):
                        continue
                    spatial_gain = 1.0 if cluster not in used_clusters else 0.0
                    depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
                    marginal = pnp_aware_marginal(
                        int(gid),
                        used_query_gain=used_query_gain,
                        cluster=cluster,
                        depth_key=depth_key,
                        spatial_gain=spatial_gain,
                        depth_gain=depth_gain,
                    ) + float(cfg.query_support_weight) * query_gain
                    next_pool.append(int(gid))
                    if best is None or float(marginal) > float(best[0]):
                        best = (float(marginal), int(gid), dict(used_query_gain), cluster, depth_key)
                if best is None:
                    break
                marginal, best_gid, used_query_gain, cluster, depth_key = best
                accept(
                    int(best_gid),
                    marginal=float(marginal),
                    used_query_gain=used_query_gain,
                    cluster=cluster,
                    depth_key=depth_key,
                )
                source_query_prefill_selected += 1
                pool = [gid for gid in next_pool if int(gid) != int(best_gid)]

    validation_query_prefill_fraction = max(0.0, float(cfg.validation_query_prefill_target_fraction))
    validation_query_prefill_max_candidates = max(0, int(cfg.validation_query_prefill_max_candidates_per_query))
    validation_query_prefill_max_per_cluster = max(0, int(cfg.validation_query_prefill_max_per_spatial_cluster))
    validation_query_prefill_min_cells = max(0, int(cfg.validation_query_prefill_min_cells_per_query))
    validation_query_prefill_min_depth_bins = max(0, int(cfg.validation_query_prefill_min_depth_bins_per_query))
    validation_query_prefill_force_all = bool(cfg.validation_query_prefill_force_all)

    def validation_query_prefill_done(query_id: str, query_target: float) -> bool:
        coverage_done = float(query_coverage.get(str(query_id), 0.0)) >= float(query_target)
        cell_done = (
            validation_query_prefill_min_cells <= 0
            or len(selected_query_cluster_counts.get(str(query_id), {})) >= validation_query_prefill_min_cells
        )
        depth_done = (
            validation_query_prefill_min_depth_bins <= 0
            or len(selected_query_depth_counts.get(str(query_id), {})) >= validation_query_prefill_min_depth_bins
        )
        return bool(coverage_done and cell_done and depth_done)

    if validation_query_prefill_fraction > 0.0 and validation_support_by_landmark:
        by_query_validation: dict[str, list[tuple[int, float]]] = {query_id: [] for query_id in query_ids}
        for gid, query_map in validation_support_by_landmark.items():
            if not bool(valid[int(gid)].item()):
                continue
            for query_id, value in query_map.items():
                raw_value = max(0.0, float(value))
                if raw_value > 0.0:
                    by_query_validation.setdefault(str(query_id), []).append((int(gid), raw_value))
        for query_id in sorted(by_query_validation):
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            candidates = by_query_validation.get(str(query_id), [])
            if not candidates:
                continue
            total_support = sum(value for _, value in candidates)
            query_target = validation_query_prefill_fraction * float(total_support)
            if query_target <= 0.0:
                continue
            ranked = sorted(
                candidates,
                key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())),
                reverse=True,
            )
            if validation_query_prefill_max_candidates > 0:
                ranked = ranked[:validation_query_prefill_max_candidates]
            pool = [int(gid) for gid, _value in ranked]
            while pool:
                if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                    break
                if (
                    not validation_query_prefill_force_all
                    and validation_query_prefill_done(str(query_id), float(query_target))
                ):
                    break
                best: tuple[float, int, dict[str, float], tuple[int, int, int], int] | None = None
                next_pool: list[int] = []
                for gid in pool:
                    ok, cluster, depth_key, _reason = can_select(int(gid))
                    if not ok:
                        continue
                    query_cluster, query_depth = _query_observation_key(
                        int(gid),
                        str(query_id),
                        observation_by_landmark,
                        fallback_cluster=cluster,
                        fallback_depth_key=depth_key,
                    )
                    query_cluster_counts = selected_query_cluster_counts.get(str(query_id), {})
                    query_depth_counts = selected_query_depth_counts.get(str(query_id), {})
                    needs_new_cell = (
                        validation_query_prefill_min_cells > 0
                        and len(query_cluster_counts) < validation_query_prefill_min_cells
                    )
                    needs_new_depth = (
                        validation_query_prefill_min_depth_bins > 0
                        and len(query_depth_counts) < validation_query_prefill_min_depth_bins
                    )
                    adds_new_cell = query_cluster_counts.get(query_cluster, 0) == 0
                    adds_new_depth = query_depth_counts.get(int(query_depth), 0) == 0
                    if (
                        validation_query_prefill_max_per_cluster > 0
                        and query_cluster_counts.get(query_cluster, 0) >= validation_query_prefill_max_per_cluster
                    ):
                        continue
                    query_gain, used_query_gain = _query_marginal_gain(
                        int(gid),
                        support_by_landmark,
                        query_coverage,
                        target_query_support=float(cfg.target_query_support),
                        easy_query_gain_decay=float(cfg.easy_query_gain_decay),
                    )
                    validation_support_value = max(
                        0.0,
                        float(validation_support_by_landmark.get(int(gid), {}).get(str(query_id), 0.0)),
                    )
                    if validation_support_value <= 0.0:
                        continue
                    used_query_gain = dict(used_query_gain)
                    used_query_gain.setdefault(str(query_id), min(validation_support_value, float(cfg.target_query_support)))
                    hard_coverage_gain = 0.0
                    if needs_new_cell and adds_new_cell:
                        hard_coverage_gain += query_prefill_hard_coverage_bonus
                    if needs_new_depth and adds_new_depth:
                        hard_coverage_gain += query_prefill_hard_coverage_bonus
                    spatial_gain = 1.0 if cluster not in used_clusters else 0.0
                    depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
                    marginal = (
                        pnp_aware_marginal(
                            int(gid),
                            used_query_gain=used_query_gain,
                            cluster=cluster,
                            depth_key=depth_key,
                            spatial_gain=spatial_gain,
                            depth_gain=depth_gain,
                        )
                        + float(cfg.query_support_weight) * query_gain
                        + 10.0 * validation_support_value
                        + hard_coverage_gain
                    )
                    next_pool.append(int(gid))
                    if best is None or float(marginal) > float(best[0]):
                        best = (float(marginal), int(gid), used_query_gain, cluster, depth_key)
                if best is None:
                    break
                marginal, best_gid, used_query_gain, cluster, depth_key = best
                accept(
                    int(best_gid),
                    marginal=float(marginal),
                    used_query_gain=used_query_gain,
                    cluster=cluster,
                    depth_key=depth_key,
                )
                validation_query_prefill_selected += 1
                pool = [gid for gid in next_pool if int(gid) != int(best_gid)]

    kc_anchor_count = max(0, int(cfg.kc_anchor_count))
    kc_anchor_max_per_cluster = max(0, int(cfg.kc_anchor_max_per_spatial_cluster))
    kc_anchor_cluster_counts: dict[tuple[int, int, int], int] = {}
    if kc_anchor_count > 0:
        anchor_candidates = [
            int(gid)
            for gid in torch.where(valid & (kc >= float(cfg.kc_anchor_min_score)))[0].tolist()
        ]
        anchor_candidates.sort(
            key=lambda gid: (
                float(kc[int(gid)].item()),
                float(visibility[int(gid)].item()),
                float(mask_validity[int(gid)].item()),
                -float(risk[int(gid)].item()),
            ),
            reverse=True,
        )
        for gid in anchor_candidates:
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            if kc_anchor_selected >= kc_anchor_count:
                break
            gate_ok, gate_reason = passes_augmentation_gate(int(gid))
            if not gate_ok:
                if gate_reason == "solver":
                    augmentation_solver_gate_rejected += 1
                elif gate_reason == "query":
                    augmentation_query_gate_rejected += 1
                elif gate_reason == "kc":
                    augmentation_kc_gate_rejected += 1
                continue
            ok, cluster, depth_key, _reason = can_select(int(gid))
            if not ok:
                continue
            if kc_anchor_max_per_cluster > 0 and kc_anchor_cluster_counts.get(cluster, 0) >= kc_anchor_max_per_cluster:
                continue
            query_gain, used_query_gain = _query_marginal_gain(
                int(gid),
                support_by_landmark,
                query_coverage,
                target_query_support=float(cfg.target_query_support),
                easy_query_gain_decay=float(cfg.easy_query_gain_decay),
            )
            spatial_gain = 1.0 if cluster not in used_clusters else 0.0
            depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
            marginal = pnp_aware_marginal(
                int(gid),
                used_query_gain=used_query_gain,
                cluster=cluster,
                depth_key=depth_key,
                spatial_gain=spatial_gain,
                depth_gain=depth_gain,
            ) + float(cfg.query_support_weight) * query_gain
            accept(
                int(gid),
                marginal=float(marginal),
                used_query_gain=used_query_gain,
                cluster=cluster,
                depth_key=depth_key,
            )
            kc_anchor_selected += 1
            kc_anchor_cluster_counts[cluster] = kc_anchor_cluster_counts.get(cluster, 0) + 1

    query_prefill_fraction = max(0.0, float(cfg.query_prefill_target_fraction))
    query_prefill_max_candidates = max(0, int(cfg.query_prefill_max_candidates_per_query))
    query_prefill_max_per_cluster = max(0, int(cfg.query_prefill_max_per_spatial_cluster))
    query_prefill_min_cells = max(0, int(cfg.query_prefill_min_cells_per_query))
    query_prefill_min_depth_bins = max(0, int(cfg.query_prefill_min_depth_bins_per_query))
    def query_prefill_done(query_id: str, query_target: float) -> bool:
        coverage_done = float(query_coverage.get(str(query_id), 0.0)) >= float(query_target)
        cell_done = (
            query_prefill_min_cells <= 0
            or len(selected_query_cluster_counts.get(str(query_id), {})) >= query_prefill_min_cells
        )
        depth_done = (
            query_prefill_min_depth_bins <= 0
            or len(selected_query_depth_counts.get(str(query_id), {})) >= query_prefill_min_depth_bins
        )
        return bool(coverage_done and cell_done and depth_done)

    if query_prefill_fraction > 0.0 and support_by_landmark:
        by_query: dict[str, list[tuple[int, float]]] = {query_id: [] for query_id in query_ids}
        for gid, query_map in support_by_landmark.items():
            if not bool(valid[int(gid)].item()):
                continue
            for query_id, value in query_map.items():
                raw_value = max(0.0, float(value))
                if raw_value > 0.0:
                    by_query.setdefault(str(query_id), []).append((int(gid), raw_value))
        query_order = sorted(
            by_query,
            key=lambda q: (
                min(float(cfg.target_query_support), sum(value for _, value in by_query.get(str(q), []))),
                str(q),
            ),
        )
        for query_id in query_order:
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            candidates = by_query.get(str(query_id), [])
            if not candidates:
                continue
            total_support = sum(value for _, value in candidates)
            query_target = query_prefill_fraction * min(float(cfg.target_query_support), float(total_support))
            if query_target <= 0.0:
                continue
            ranked = sorted(
                candidates,
                key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())),
                reverse=True,
            )
            if query_prefill_max_candidates > 0:
                ranked = ranked[:query_prefill_max_candidates]
            pool = [int(gid) for gid, _raw_value in ranked]
            while pool:
                if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                    break
                if query_prefill_done(str(query_id), float(query_target)):
                    break
                best: tuple[float, int, dict[str, float], tuple[int, int, int], int] | None = None
                next_pool: list[int] = []
                for gid in pool:
                    gate_ok, gate_reason = passes_augmentation_gate(int(gid))
                    if not gate_ok:
                        if gate_reason == "solver":
                            augmentation_solver_gate_rejected += 1
                        elif gate_reason == "query":
                            augmentation_query_gate_rejected += 1
                        elif gate_reason == "kc":
                            augmentation_kc_gate_rejected += 1
                        continue
                    ok, cluster, depth_key, _reason = can_select(int(gid))
                    if not ok:
                        continue
                    query_cluster, query_depth = _query_observation_key(
                        int(gid),
                        str(query_id),
                        observation_by_landmark,
                        fallback_cluster=cluster,
                        fallback_depth_key=depth_key,
                    )
                    query_cluster_counts = selected_query_cluster_counts.get(str(query_id), {})
                    query_depth_counts = selected_query_depth_counts.get(str(query_id), {})
                    needs_new_cell = query_prefill_min_cells > 0 and len(query_cluster_counts) < query_prefill_min_cells
                    needs_new_depth = (
                        query_prefill_min_depth_bins > 0 and len(query_depth_counts) < query_prefill_min_depth_bins
                    )
                    adds_new_cell = query_cluster_counts.get(query_cluster, 0) == 0
                    adds_new_depth = query_depth_counts.get(int(query_depth), 0) == 0
                    if (
                        query_prefill_max_per_cluster > 0
                        and query_cluster_counts.get(query_cluster, 0) >= query_prefill_max_per_cluster
                    ):
                        continue
                    query_gain, used_query_gain = _query_marginal_gain(
                        int(gid),
                        support_by_landmark,
                        query_coverage,
                        target_query_support=float(cfg.target_query_support),
                        easy_query_gain_decay=float(cfg.easy_query_gain_decay),
                    )
                    hard_coverage_gain = 0.0
                    if needs_new_cell and adds_new_cell:
                        hard_coverage_gain += query_prefill_hard_coverage_bonus
                    if needs_new_depth and adds_new_depth:
                        hard_coverage_gain += query_prefill_hard_coverage_bonus
                    if hard_coverage_gain > 0.0 and str(query_id) in support_by_landmark.get(int(gid), {}):
                        used_query_gain = dict(used_query_gain)
                        used_query_gain.setdefault(str(query_id), 1.0e-6)
                    if query_gain <= 0.0 and hard_coverage_gain <= 0.0:
                        continue
                    spatial_gain = 1.0 if cluster not in used_clusters else 0.0
                    depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
                    marginal = pnp_aware_marginal(
                        int(gid),
                        used_query_gain=used_query_gain,
                        cluster=cluster,
                        depth_key=depth_key,
                        spatial_gain=spatial_gain,
                        depth_gain=depth_gain,
                    ) + float(cfg.query_support_weight) * query_gain + hard_coverage_gain
                    next_pool.append(int(gid))
                    if best is None or float(marginal) > float(best[0]):
                        best = (float(marginal), int(gid), used_query_gain, cluster, depth_key)
                if best is None:
                    break
                marginal, best_gid, used_query_gain, cluster, depth_key = best
                accept(
                    int(best_gid),
                    marginal=float(marginal),
                    used_query_gain=used_query_gain,
                    cluster=cluster,
                    depth_key=depth_key,
                )
                query_prefill_selected += 1
                pool = [gid for gid in next_pool if int(gid) != int(best_gid)]

    main_dynamic_evaluation_count = 0

    def evaluate_main_candidate(
        gid: int,
    ) -> tuple[tuple[float, int, dict[str, float], tuple[int, int, int], int] | None, str]:
        nonlocal main_dynamic_evaluation_count
        nonlocal main_query_match_context_gate_accepted
        main_dynamic_evaluation_count += 1
        gate_ok, gate_reason = passes_augmentation_gate(int(gid))
        if not gate_ok:
            return None, f"augmentation_{gate_reason}"
        ok, cluster, depth_key, reason = can_select(int(gid))
        if not ok:
            return None, reason

        query_gain, used_query_gain = _query_marginal_gain(
            int(gid),
            support_by_landmark,
            query_coverage,
            target_query_support=float(cfg.target_query_support),
            easy_query_gain_decay=float(cfg.easy_query_gain_decay),
        )
        match_strength_gain, _match_risk, _match_ratio, _match_reference, _match_nonreference = (
            query_match_dominance_delta(int(gid))
        )
        weighted_context_gate_gain = (
            max(0.0, float(cfg.query_match_strength_weight)) * max(0.0, float(match_strength_gain))
        )
        source_gate_gain = soft_source_anchor_gate_gain(int(gid))
        if (
            len(selected) >= min_landmarks
            and float(cfg.main_min_query_gain) > 0.0
            and float(query_gain) + source_gate_gain + weighted_context_gate_gain < float(cfg.main_min_query_gain)
        ):
            return None, "main_query_gain"
        if (
            len(selected) >= min_landmarks
            and float(cfg.main_min_query_gain) > 0.0
            and float(query_gain) + source_gate_gain < float(cfg.main_min_query_gain)
            and weighted_context_gate_gain > 0.0
        ):
            main_query_match_context_gate_accepted += 1
        spatial_gain = 1.0 if cluster not in used_clusters else 0.0
        depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
        pnp_bootstrap_gain = 0.0
        for query_id in support_by_landmark.get(int(gid), {}):
            count = int(selected_query_counts.get(str(query_id), 0))
            if count == 0:
                pnp_bootstrap_gain += 0.5
            elif count < 4:
                pnp_bootstrap_gain += 0.25

        marginal = pnp_aware_marginal(
            int(gid),
            used_query_gain=used_query_gain,
            cluster=cluster,
            depth_key=depth_key,
            spatial_gain=spatial_gain,
            depth_gain=depth_gain,
            pnp_bootstrap_gain=pnp_bootstrap_gain,
        ) + float(cfg.query_support_weight) * query_gain

        if len(selected) >= min_landmarks and marginal < float(cfg.min_marginal_gain):
            return None, "low_gain"

        return (float(marginal), int(gid), dict(used_query_gain), cluster, depth_key), ""

    def count_main_rejection(reason: str) -> None:
        nonlocal augmentation_solver_gate_rejected
        nonlocal augmentation_query_gate_rejected
        nonlocal augmentation_kc_gate_rejected
        nonlocal cluster_cap_rejected
        nonlocal low_gain_rejected
        nonlocal main_query_gain_gate_rejected
        if reason == "augmentation_solver":
            augmentation_solver_gate_rejected += 1
        elif reason == "augmentation_query":
            augmentation_query_gate_rejected += 1
        elif reason == "augmentation_kc":
            augmentation_kc_gate_rejected += 1
        elif reason == "cluster_cap":
            cluster_cap_rejected += 1
        elif reason == "low_gain":
            low_gain_rejected += 1
        elif reason == "main_query_gain":
            main_query_gain_gate_rejected += 1

    main_dynamic_lookahead = max(0, int(cfg.main_dynamic_lookahead))
    if main_dynamic_lookahead <= 1:
        # A zero/one lookahead used to degrade into static-order acceptance.  That
        # is not a set objective: an early high unary/K.C. candidate could consume
        # budget before a later candidate with much higher current PnP support.
        # Keep the auto window bounded so full-Gaussian selection remains tractable.
        main_dynamic_effective_lookahead = min(len(ordered_ids), 4096)
    else:
        main_dynamic_effective_lookahead = min(len(ordered_ids), main_dynamic_lookahead)

    if main_dynamic_effective_lookahead <= 1:
        for gid in ordered_ids:
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            evaluation, reason = evaluate_main_candidate(int(gid))
            if evaluation is None:
                count_main_rejection(reason)
                continue
            marginal, eval_gid, used_query_gain, cluster, depth_key = evaluation
            accept(
                int(eval_gid),
                marginal=float(marginal),
                used_query_gain=used_query_gain,
                cluster=cluster,
                depth_key=depth_key,
            )
    else:
        cursor = 0
        pending: list[int] = []
        while True:
            if pre_local_max_landmarks is not None and len(selected) >= pre_local_max_landmarks:
                break
            while len(pending) < main_dynamic_effective_lookahead and cursor < len(ordered_ids):
                gid = int(ordered_ids[cursor])
                cursor += 1
                if gid not in selected_set:
                    pending.append(gid)
            if not pending:
                break
            best: tuple[float, int, dict[str, float], tuple[int, int, int], int] | None = None
            next_pending: list[int] = []
            for gid in pending:
                evaluation, reason = evaluate_main_candidate(int(gid))
                if evaluation is None:
                    count_main_rejection(reason)
                    continue
                next_pending.append(int(gid))
                if best is None or float(evaluation[0]) > float(best[0]):
                    best = evaluation
            if best is None:
                pending = []
                continue
            marginal, eval_gid, used_query_gain, cluster, depth_key = best
            accept(
                int(eval_gid),
                marginal=float(marginal),
                used_query_gain=used_query_gain,
                cluster=cluster,
                depth_key=depth_key,
            )
            pending = [int(gid) for gid in next_pending if int(gid) != int(eval_gid)]

    local_search_added_count = 0
    local_search_failed_query_round_count = 0
    if local_search_rounds > 0 and local_search_candidates_per_query > 0 and support_by_landmark:
        support_by_query: dict[str, list[tuple[int, float]]] = {query_id: [] for query_id in query_ids}
        for gid, query_map in support_by_landmark.items():
            if not bool(valid[int(gid)].item()):
                continue
            for query_id, value in query_map.items():
                if float(value) > 0.0:
                    support_by_query.setdefault(str(query_id), []).append((int(gid), float(value)))
        for query_id in support_by_query:
            support_by_query[query_id].sort(
                key=lambda item: (float(item[1]), float(static_score[int(item[0])].item())),
                reverse=True,
            )

        for _round_idx in range(local_search_rounds):
            failed_queries, _validation_metrics = _query_validation_failures(
                selected,
                query_ids,
                support_by_landmark,
                observation_by_landmark,
                xyz,
                cfg,
            )
            if not failed_queries:
                break
            local_search_failed_query_round_count += len(failed_queries)
            made_progress = False
            for query_id in sorted(failed_queries):
                if max_landmarks is not None and len(selected) >= max_landmarks:
                    break
                if local_search_max_additions > 0 and local_search_added_count >= local_search_max_additions:
                    break
                raw_candidates = support_by_query.get(str(query_id), [])[:local_search_candidates_per_query]
                best: tuple[float, int, dict[str, float], tuple[int, int, int], int] | None = None
                for gid, support_value in raw_candidates:
                    gate_ok, gate_reason = passes_augmentation_gate(int(gid))
                    if not gate_ok:
                        if gate_reason == "solver":
                            augmentation_solver_gate_rejected += 1
                        elif gate_reason == "query":
                            augmentation_query_gate_rejected += 1
                        elif gate_reason == "kc":
                            augmentation_kc_gate_rejected += 1
                        continue
                    ok, cluster, depth_key, _reason = can_select(int(gid))
                    if not ok:
                        continue
                    query_gain, used_query_gain = _query_marginal_gain(
                        int(gid),
                        support_by_landmark,
                        query_coverage,
                        target_query_support=float(cfg.target_query_support),
                        easy_query_gain_decay=float(cfg.easy_query_gain_decay),
                    )
                    if query_gain <= 0.0 and str(query_id) not in support_by_landmark.get(int(gid), {}):
                        continue
                    if str(query_id) in support_by_landmark.get(int(gid), {}) and str(query_id) not in used_query_gain:
                        used_query_gain = dict(used_query_gain)
                        used_query_gain[str(query_id)] = min(1.0e-6, max(1.0e-6, float(support_value)))
                    validation_gain = _validation_gain_for_query(
                        int(gid),
                        str(query_id),
                        selected,
                        support_by_landmark,
                        observation_by_landmark,
                        xyz,
                        cfg,
                    )
                    if validation_gain <= 0.0:
                        continue
                    spatial_gain = 1.0 if cluster not in used_clusters else 0.0
                    depth_gain = 1.0 if depth_key not in used_depth_bins else 0.0
                    marginal = (
                        pnp_aware_marginal(
                            int(gid),
                            used_query_gain=used_query_gain,
                            cluster=cluster,
                            depth_key=depth_key,
                            spatial_gain=spatial_gain,
                            depth_gain=depth_gain,
                        )
                        + float(cfg.query_support_weight) * query_gain
                        + 10.0 * validation_gain
                    )
                    if best is None or float(marginal) > float(best[0]):
                        best = (float(marginal), int(gid), dict(used_query_gain), cluster, depth_key)
                if best is None:
                    continue
                marginal, best_gid, used_query_gain, cluster, depth_key = best
                accept(
                    int(best_gid),
                    marginal=float(marginal),
                    used_query_gain=used_query_gain,
                    cluster=cluster,
                    depth_key=depth_key,
                )
                local_search_added_count += 1
                made_progress = True
            if not made_progress:
                break

    precision_fill_selected_count = 0
    precision_fill_low_score_rejected_count = 0
    precision_fill_no_query_support_rejected_count = 0
    precision_fill_cluster_rejected_count = 0
    precision_fill_target = max(0, int(cfg.precision_fill_target_count))
    if max_landmarks is not None and precision_fill_target > 0:
        precision_fill_target = min(int(precision_fill_target), int(max_landmarks))
    precision_fill_max_per_cluster = max(0, int(cfg.precision_fill_max_per_spatial_cluster))
    precision_fill_min_score = float(cfg.precision_fill_min_score)
    if precision_fill_target > len(selected):
        precision_fill_score = (
            float(cfg.precision_fill_kc_weight) * _normalize(kc)
            + float(cfg.precision_fill_visibility_weight) * _normalize(visibility)
            + float(cfg.precision_fill_mask_weight) * _normalize(mask_validity)
            + float(cfg.precision_fill_solver_weight) * _normalize(solver)
            + float(cfg.precision_fill_query_support_weight) * _normalize(query_raw_support)
            + float(cfg.precision_fill_support_match_strength_weight) * _normalize(support_match_strength)
            - float(cfg.precision_fill_ambiguity_risk_weight) * _normalize(risk)
            - float(cfg.precision_fill_sparse_validation_risk_weight) * _normalize(sparse_validation_risk)
            - float(cfg.precision_fill_match_competition_risk_weight) * _normalize(match_competition_risk)
        )
        precision_fill_score = torch.where(valid, precision_fill_score, torch.full_like(precision_fill_score, -float("inf")))
        precision_order = torch.argsort(precision_fill_score[candidate_ids], descending=True)
        for raw_pos in precision_order.tolist():
            if len(selected) >= precision_fill_target:
                break
            gid = int(candidate_ids[int(raw_pos)].item())
            if bool(cfg.precision_fill_require_query_support) and int(gid) not in support_by_landmark:
                precision_fill_no_query_support_rejected_count += 1
                continue
            ok, cluster, depth_key, reason = can_select(int(gid))
            if not ok:
                if reason == "cluster_cap":
                    precision_fill_cluster_rejected_count += 1
                continue
            if precision_fill_max_per_cluster > 0 and cluster_counts.get(cluster, 0) >= precision_fill_max_per_cluster:
                precision_fill_cluster_rejected_count += 1
                continue
            conflict_penalty = _descriptor_conflict_penalty(int(gid), selected_set, descriptor_conflict_edges)
            score = float(precision_fill_score[int(gid)].item()) - float(cfg.precision_fill_descriptor_conflict_weight) * conflict_penalty
            if not math.isfinite(score) or score < precision_fill_min_score:
                precision_fill_low_score_rejected_count += 1
                continue
            query_gain, used_query_gain = _query_marginal_gain(
                int(gid),
                support_by_landmark,
                query_coverage,
                target_query_support=float(cfg.target_query_support),
                easy_query_gain_decay=float(cfg.easy_query_gain_decay),
            )
            del query_gain
            accept(
                int(gid),
                marginal=float(score),
                used_query_gain=used_query_gain,
                cluster=cluster,
                depth_key=depth_key,
            )
            precision_fill_selected_count += 1

    final_prune_min_keep = max(0, int(cfg.final_prune_min_keep_count))
    local_pruned_conflict_count = 0
    local_prune_conflict_threshold = max(0.0, float(cfg.local_prune_conflict_threshold))
    if selected and local_prune_conflict_threshold > 0.0 and descriptor_conflict_edges:
        keep_set = set(int(gid) for gid in selected)
        local_prune_validate_queries = bool(cfg.local_prune_validate_queries)
        hard_source_protected = source_anchor_mode == "hard"

        def protected_from_local_prune(gid: int) -> bool:
            return bool(
                (hard_source_protected and int(gid) in source_anchor_set)
                or is_sparse_validation_protected(int(gid))
            )

        def local_conflict_total(gid: int, pool: set[int]) -> float:
            neighbors = descriptor_conflict_edges.get(int(gid), {})
            return float(
                sum(
                    max(0.0, float(value))
                    for raw_other, value in neighbors.items()
                    if int(raw_other) in pool and int(raw_other) != int(gid)
                )
            )

        def removal_preserves_validation(gid: int, pool: set[int]) -> bool:
            if not local_prune_validate_queries:
                return True
            next_pool = set(pool)
            next_pool.discard(int(gid))
            failed, _metrics = _query_validation_failures(
                next_pool,
                query_ids,
                support_by_landmark,
                observation_by_landmark,
                xyz,
                cfg,
            )
            return not failed

        prune_order = sorted(
            [int(gid) for gid in keep_set if not protected_from_local_prune(int(gid))],
            key=lambda gid: (
                local_conflict_total(int(gid), keep_set),
                -float(query_raw_support[int(gid)].item()),
                int(gid),
            ),
            reverse=True,
        )
        for gid in prune_order:
            if len(keep_set) <= final_prune_min_keep:
                break
            if gid not in keep_set:
                continue
            if local_conflict_total(int(gid), keep_set) <= local_prune_conflict_threshold:
                continue
            if not removal_preserves_validation(int(gid), keep_set):
                continue
            keep_set.remove(int(gid))
            local_pruned_conflict_count += 1
        selected = [int(gid) for gid in selected if int(gid) in keep_set]

    final_pruned_no_query_utility_count = 0
    final_pruned_conflict_count = 0
    final_pruned_query_match_risk_reference_count = 0
    final_pruned_nonreference_query_match_competition_count = 0
    final_prune_conflict_threshold = max(0.0, float(cfg.final_prune_conflict_threshold))
    if selected and (
        bool(cfg.final_prune_no_query_utility)
        or (final_prune_conflict_threshold > 0.0 and descriptor_conflict_edges)
    ):
        keep_set = set(int(gid) for gid in selected)
        hard_source_protected = source_anchor_mode == "hard"

        def is_protected(gid: int) -> bool:
            return bool(
                (hard_source_protected and int(gid) in source_anchor_set)
                or is_sparse_validation_protected(int(gid))
            )

        def selected_utility(gid: int) -> float:
            return float(
                max(0.0, float(query_raw_support[int(gid)].item()))
                + max(0.0, float(solver[int(gid)].item()))
                + max(0.0, float(static_score[int(gid)].item()))
            )

        prune_order = sorted(
            [int(gid) for gid in selected if not is_protected(int(gid))],
            key=lambda gid: (selected_utility(int(gid)), int(gid)),
        )
        for gid in prune_order:
            if len(keep_set) <= final_prune_min_keep:
                break
            if gid not in keep_set:
                continue
            no_query_utility = bool(cfg.final_prune_no_query_utility) and int(gid) not in support_by_landmark
            conflict_total = sum(
                max(0.0, float(value))
                for raw_other, value in descriptor_conflict_edges.get(int(gid), {}).items()
                if int(raw_other) in keep_set and int(raw_other) != int(gid)
            )
            conflict_bad = final_prune_conflict_threshold > 0.0 and conflict_total > final_prune_conflict_threshold
            if not no_query_utility and not conflict_bad:
                continue
            keep_set.remove(int(gid))
            if no_query_utility:
                final_pruned_no_query_utility_count += 1
            elif conflict_bad:
                final_pruned_conflict_count += 1
        selected = [int(gid) for gid in selected if int(gid) in keep_set]

    if selected and bool(cfg.final_prune_nonreference_query_match_competition) and query_match_competition_reference_landmarks:
        keep_set = set(int(gid) for gid in selected)
        hard_source_protected = source_anchor_mode == "hard"

        def is_nonreference_prune_protected(gid: int) -> bool:
            return bool(
                (hard_source_protected and int(gid) in source_anchor_set)
                or is_sparse_validation_protected(int(gid))
            )

        def nonreference_competition_score(gid: int) -> float:
            score = 0.0
            for raw_query_id, raw_value in query_match_risk_by_landmark.get(int(gid), {}).items():
                query_key = str(raw_query_id)
                reference_landmarks = query_match_competition_reference_landmarks.get(query_key)
                if reference_landmarks is None or int(gid) in reference_landmarks:
                    continue
                score += max(0.0, float(raw_value))
            return float(score)

        def support_utility_for_nonreference_prune(gid: int) -> float:
            return float(
                max(0.0, float(query_raw_support[int(gid)].item()))
                + sum(
                    max(0.0, float(value))
                    for value in query_match_strength_by_landmark.get(int(gid), {}).values()
                )
                + max(0.0, float(solver[int(gid)].item()))
            )

        prune_order = sorted(
            [int(gid) for gid in keep_set if not is_nonreference_prune_protected(int(gid))],
            key=lambda gid: (
                nonreference_competition_score(int(gid)),
                -support_utility_for_nonreference_prune(int(gid)),
                -float(static_score[int(gid)].item()),
                int(gid),
            ),
            reverse=True,
        )
        for gid in prune_order:
            if len(keep_set) <= final_prune_min_keep:
                break
            if gid not in keep_set:
                continue
            if nonreference_competition_score(int(gid)) <= 0.0:
                break
            keep_set.remove(int(gid))
            final_pruned_nonreference_query_match_competition_count += 1
        selected = [int(gid) for gid in selected if int(gid) in keep_set]

    query_match_risk_reference_margin = max(0.0, float(cfg.query_match_risk_reference_margin))
    if selected and bool(cfg.final_prune_query_match_risk_reference) and query_match_risk_reference:
        keep_set = set(int(gid) for gid in selected)
        hard_source_protected = source_anchor_mode == "hard"

        def is_reference_prune_protected(gid: int) -> bool:
            return bool(
                (hard_source_protected and int(gid) in source_anchor_set)
                or is_sparse_validation_protected(int(gid))
            )

        def reference_cap(query_id: str) -> float:
            return max(0.0, float(query_match_risk_reference[str(query_id)])) + query_match_risk_reference_margin

        def query_risk_for_pool(pool: set[int]) -> dict[str, float]:
            out = {query_id: 0.0 for query_id in query_ids}
            for pool_gid in pool:
                for raw_query_id, raw_value in query_match_risk_by_landmark.get(int(pool_gid), {}).items():
                    query_key = str(raw_query_id)
                    out[query_key] = float(out.get(query_key, 0.0) + max(0.0, float(raw_value)))
            return out

        def risk_over_reference(pool: set[int]) -> dict[str, float]:
            risks = query_risk_for_pool(pool)
            return {
                str(query_id): max(0.0, float(risks.get(str(query_id), 0.0)) - reference_cap(str(query_id)))
                for query_id in query_match_risk_reference.keys()
            }

        def support_utility(gid: int) -> float:
            return float(
                max(0.0, float(query_raw_support[int(gid)].item()))
                + sum(
                    max(0.0, float(value))
                    for value in query_match_strength_by_landmark.get(int(gid), {}).values()
                )
                + max(0.0, float(solver[int(gid)].item()))
            )

        while len(keep_set) > final_prune_min_keep:
            over_by_query = risk_over_reference(keep_set)
            if not any(float(value) > 0.0 for value in over_by_query.values()):
                break
            best: tuple[float, float, float, int] | None = None
            for gid in list(keep_set):
                if is_reference_prune_protected(int(gid)):
                    continue
                contribution = 0.0
                for raw_query_id, raw_value in query_match_risk_by_landmark.get(int(gid), {}).items():
                    query_key = str(raw_query_id)
                    over = max(0.0, float(over_by_query.get(query_key, 0.0)))
                    if over <= 0.0:
                        continue
                    contribution += min(max(0.0, float(raw_value)), over)
                if contribution <= 0.0:
                    continue
                utility = support_utility(int(gid))
                static = float(static_score[int(gid)].item())
                candidate = (float(contribution), -float(utility), -float(static), int(gid))
                if best is None or candidate > best:
                    best = candidate
            if best is None:
                break
            keep_set.remove(int(best[3]))
            final_pruned_query_match_risk_reference_count += 1
        selected = [int(gid) for gid in selected if int(gid) in keep_set]

    sampled = torch.tensor(selected, dtype=torch.long)
    target = max(1.0e-6, float(cfg.target_query_support))
    final_query_coverage = {query_id: 0.0 for query_id in query_ids}
    final_query_match_strength = {query_id: 0.0 for query_id in query_ids}
    final_query_match_risk = {query_id: 0.0 for query_id in query_ids}
    final_query_cluster_counts: dict[str, dict[tuple[int, int, int], int]] = {query_id: {} for query_id in query_ids}
    final_query_depth_counts: dict[str, dict[int, int]] = {query_id: {} for query_id in query_ids}
    final_used_clusters: set[tuple[int, int, int]] = set()
    final_used_depth_bins: set[int] = set()
    for gid in selected:
        point = xyz[int(gid)]
        cluster = _cluster_key(point, float(cfg.spatial_cluster_size_m))
        depth_key = _depth_bin(point, float(cfg.depth_bin_size_m))
        final_used_clusters.add(cluster)
        final_used_depth_bins.add(depth_key)
        _query_gain, used_query_gain = _query_marginal_gain(
            int(gid),
            support_by_landmark,
            final_query_coverage,
            target_query_support=float(cfg.target_query_support),
            easy_query_gain_decay=float(cfg.easy_query_gain_decay),
        )
        for query_id, value in used_query_gain.items():
            query_key = str(query_id)
            query_cluster, query_depth = _query_observation_key(
                int(gid),
                query_key,
                observation_by_landmark,
                fallback_cluster=cluster,
                fallback_depth_key=depth_key,
            )
            final_query_coverage[query_key] = float(final_query_coverage.get(query_key, 0.0)) + max(0.0, float(value))
            per_query_clusters = final_query_cluster_counts.setdefault(query_key, {})
            per_query_clusters[query_cluster] = per_query_clusters.get(query_cluster, 0) + 1
            per_query_depths = final_query_depth_counts.setdefault(query_key, {})
            per_query_depths[int(query_depth)] = per_query_depths.get(int(query_depth), 0) + 1
        for query_id, value in query_match_strength_by_landmark.get(int(gid), {}).items():
            query_key = str(query_id)
            final_query_match_strength[query_key] = float(
                final_query_match_strength.get(query_key, 0.0) + max(0.0, float(value))
            )
        for query_id, value in query_match_risk_by_landmark.get(int(gid), {}).items():
            query_key = str(query_id)
            final_query_match_risk[query_key] = float(
                final_query_match_risk.get(query_key, 0.0) + max(0.0, float(value))
            )
    unmet_query_count = sum(1 for value in final_query_coverage.values() if float(value) < target)
    final_query_cell_counts = {
        query_id: int(len(final_query_cluster_counts.get(query_id, {}))) for query_id in query_ids
    }
    final_query_depth_bin_counts = {
        query_id: int(len(final_query_depth_counts.get(query_id, {}))) for query_id in query_ids
    }
    final_query_match_risk_over_reference = {
        str(query_id): max(
            0.0,
            float(final_query_match_risk.get(str(query_id), 0.0))
            - (max(0.0, float(reference_value)) + query_match_risk_reference_margin),
        )
        for query_id, reference_value in query_match_risk_reference.items()
    }
    final_query_match_risk_over_reference_count = int(
        sum(1 for value in final_query_match_risk_over_reference.values() if float(value) > 0.0)
    )
    final_query_match_nonreference_competition = {query_id: 0.0 for query_id in query_match_competition_reference_landmarks}
    for gid in selected:
        for raw_query_id, raw_value in query_match_risk_by_landmark.get(int(gid), {}).items():
            query_key = str(raw_query_id)
            reference_landmarks = query_match_competition_reference_landmarks.get(query_key)
            if reference_landmarks is None or int(gid) in reference_landmarks:
                continue
            final_query_match_nonreference_competition[query_key] = float(
                final_query_match_nonreference_competition.get(query_key, 0.0)
                + max(0.0, float(raw_value))
            )
    final_query_match_nonreference_competition_count = int(
        sum(1 for value in final_query_match_nonreference_competition.values() if float(value) > 0.0)
    )
    validation_failed_queries, validation_query_metrics = _query_validation_failures(
        selected,
        query_ids,
        support_by_landmark,
        observation_by_landmark,
        xyz,
        cfg,
    )
    selected_signal_stats = {
        "kc_score": _signal_stats_for_ids(kc, selected),
        "visibility_score": _signal_stats_for_ids(visibility, selected),
        "mask_validity": _signal_stats_for_ids(mask_validity, selected),
        "solver_support": _signal_stats_for_ids(solver, selected),
        "query_raw_support": _signal_stats_for_ids(query_raw_support, selected),
        "ambiguity_risk": _signal_stats_for_ids(risk, selected),
        "sparse_validation_risk": _signal_stats_for_ids(sparse_validation_risk, selected),
        "support_match_strength": _signal_stats_for_ids(support_match_strength, selected),
        "match_competition_risk": _signal_stats_for_ids(match_competition_risk, selected),
        "static_score": _signal_stats_for_ids(static_score, selected),
    }
    metadata = {
        "selection_policy": "full_gaussian_sparse_set",
        "source": "full_gaussians",
        "budget_mode": "variable_size_marginal_gain",
        "input_gaussian_count": int(size),
        "candidate_count": int(candidate_ids.numel()),
        "candidate_pool_static_count": int(candidate_pool_static_count),
        "candidate_pool_source_anchor_added_count": int(candidate_pool_source_anchor_added_count),
        "candidate_pool_kc_added_count": int(candidate_pool_kc_added_count),
        "candidate_pool_visibility_added_count": int(candidate_pool_visibility_added_count),
        "candidate_pool_mask_added_count": int(candidate_pool_mask_added_count),
        "candidate_pool_solver_added_count": int(candidate_pool_solver_added_count),
        "candidate_pool_match_strength_added_count": int(candidate_pool_match_strength_added_count),
        "candidate_pool_query_match_strength_added_count": int(candidate_pool_query_match_strength_added_count),
        "candidate_pool_kc_top_k": int(candidate_pool_kc_top_k),
        "candidate_pool_visibility_top_k": int(candidate_pool_visibility_top_k),
        "candidate_pool_mask_top_k": int(candidate_pool_mask_top_k),
        "candidate_pool_solver_top_k": int(candidate_pool_solver_top_k),
        "candidate_pool_match_strength_top_k": int(candidate_pool_match_strength_top_k),
        "candidate_pool_query_match_strength_top_k": int(query_match_strength_top_k),
        "candidate_pool_query_added_count": int(candidate_pool_query_added_count),
        "candidate_pool_query_top_k": int(query_top_k),
        "candidate_pool_query_cell_top_k": int(query_cell_top_k),
        "candidate_pool_query_depth_top_k": int(query_depth_top_k),
        "selected_count": int(sampled.numel()),
        "selected_signal_stats": selected_signal_stats,
        "max_landmarks": int(max_landmarks) if max_landmarks is not None else None,
        "main_selection_cap": int(pre_local_max_landmarks)
        if pre_local_max_landmarks is not None
        else None,
        "min_landmarks": int(min_landmarks),
        "min_marginal_gain": float(cfg.min_marginal_gain),
        "query_count": int(len(query_ids)),
        "query_evidence_landmark_count": int(len(support_by_landmark)),
        "validation_query_evidence_landmark_count": int(len(validation_support_by_landmark)),
        "validation_query_evidence_query_count": int(len(validation_query_ids)),
        "query_observation_landmark_count": int(len(observation_by_landmark)),
        "query_observation_entry_count": int(sum(len(query_map) for query_map in observation_by_landmark.values())),
        "final_query_coverage": final_query_coverage,
        "final_query_match_strength": final_query_match_strength,
        "final_query_match_risk": final_query_match_risk,
        "query_match_risk_reference": query_match_risk_reference,
        "query_match_risk_reference_margin": float(query_match_risk_reference_margin),
        "query_match_risk_reference_weight": float(cfg.query_match_risk_reference_weight),
        "query_match_risk_reference_query_count": int(len(query_match_risk_reference)),
        "final_query_match_risk_over_reference": final_query_match_risk_over_reference,
        "final_query_match_risk_over_reference_count": int(final_query_match_risk_over_reference_count),
        "query_match_competition_reference_query_count": int(len(query_match_competition_reference_landmarks)),
        "query_match_competition_reference_entry_count": int(
            sum(len(values) for values in query_match_competition_reference_landmarks.values())
        ),
        "query_match_nonreference_competition_weight": float(
            cfg.query_match_nonreference_competition_weight
        ),
        "final_query_match_nonreference_competition": final_query_match_nonreference_competition,
        "final_query_match_nonreference_competition_count": int(
            final_query_match_nonreference_competition_count
        ),
        "query_match_strength_weight": float(cfg.query_match_strength_weight),
        "query_match_competition_weight": float(cfg.query_match_competition_weight),
        "query_match_competition_ratio_weight": float(cfg.query_match_competition_ratio_weight),
        "query_match_strength_query_count": int(len(query_match_strength_by_query)),
        "query_match_strength_landmark_count": int(len(query_match_strength_by_landmark)),
        "query_match_strength_entry_count": int(
            sum(len(query_map) for query_map in query_match_strength_by_landmark.values())
        ),
        "query_match_competition_risk_query_count": int(len(query_match_risk_by_query)),
        "query_match_competition_risk_landmark_count": int(len(query_match_risk_by_landmark)),
        "query_match_competition_risk_entry_count": int(
            sum(len(query_map) for query_map in query_match_risk_by_landmark.values())
        ),
        "support_match_strength_weight": float(cfg.support_match_strength_weight),
        "support_match_strength_nonzero_count": int((support_match_strength > 0).sum().item()),
        "support_match_strength_total": float(support_match_strength.sum().item()),
        "support_match_strength_max": float(support_match_strength.max().item())
        if int(support_match_strength.numel()) > 0
        else 0.0,
        "match_competition_risk_weight": float(cfg.match_competition_risk_weight),
        "match_competition_risk_reject_threshold": float(match_competition_risk_threshold),
        "match_competition_risk_nonzero_count": int((match_competition_risk > 0).sum().item()),
        "match_competition_risk_max": float(match_competition_risk.max().item())
        if int(match_competition_risk.numel()) > 0
        else 0.0,
        "match_competition_risk_hard_rejected_count": int(
            (finite & mask_ok & visibility_ok & sparse_validation_risk_ok & ~match_competition_risk_ok).sum().item()
        ),
        "descriptor_conflict_edge_count": int(descriptor_conflict_edge_count),
        "explicit_descriptor_conflict_edge_count": int(
            sum(len(neighbors) for neighbors in explicit_descriptor_conflict_edges.values())
        ),
        "auto_descriptor_conflict_edge_count": int(auto_descriptor_conflict_edge_count),
        "descriptor_conflict_weight": float(cfg.descriptor_conflict_weight),
        "auto_descriptor_conflict_top_k": int(cfg.auto_descriptor_conflict_top_k),
        "auto_descriptor_conflict_candidate_limit": int(cfg.auto_descriptor_conflict_candidate_limit),
        "auto_descriptor_conflict_min_cosine": float(cfg.auto_descriptor_conflict_min_cosine),
        "auto_descriptor_conflict_min_spatial_distance_m": float(
            cfg.auto_descriptor_conflict_min_spatial_distance_m
        ),
        "sparse_validation_risk_weight": float(cfg.sparse_validation_risk_weight),
        "sparse_validation_risk_reject_threshold": float(sparse_validation_risk_threshold),
        "sparse_validation_protected_hard_reject_exempt": bool(
            sparse_validation_risk_protected_hard_reject_exempt
        ),
        "sparse_validation_risk_source_anchor_exempt": bool(
            sparse_validation_risk_source_anchor_exempt
        ),
        "sparse_validation_risk_source_anchor_exempt_count": int(
            sparse_validation_risk_source_anchor_exempt_count
        ),
        "sparse_validation_prune_protected_count": int(sparse_validation_prune_protected_count),
        "sparse_validation_hard_reject_exempt_count": int(sparse_validation_hard_reject_exempt_count),
        "sparse_validation_risk_hard_reject_exempt_count": int(
            sparse_validation_risk_hard_reject_exempt_count
        ),
        "sparse_validation_risk_original_nonzero_count": int(sparse_validation_risk_original_nonzero_count),
        "sparse_validation_risk_nonzero_count": int((sparse_validation_risk > 0).sum().item()),
        "sparse_validation_risk_max": float(sparse_validation_risk.max().item())
        if int(sparse_validation_risk.numel()) > 0
        else 0.0,
        "sparse_validation_risk_hard_rejected_count": int(
            (finite & mask_ok & visibility_ok & ~sparse_validation_risk_ok).sum().item()
        ),
        "sparse_validation_risk_expand_top_k": int(cfg.sparse_validation_risk_expand_top_k),
        "sparse_validation_risk_expand_neighbors_per_seed": int(
            cfg.sparse_validation_risk_expand_neighbors_per_seed
        ),
        "sparse_validation_risk_expand_candidate_limit": int(cfg.sparse_validation_risk_expand_candidate_limit),
        "sparse_validation_risk_expand_min_cosine": float(cfg.sparse_validation_risk_expand_min_cosine),
        "sparse_validation_risk_expand_min_spatial_distance_m": float(
            cfg.sparse_validation_risk_expand_min_spatial_distance_m
        ),
        "sparse_validation_risk_expand_weight": float(cfg.sparse_validation_risk_expand_weight),
        **sparse_validation_risk_expansion_metadata,
        "final_prune_no_query_utility": bool(cfg.final_prune_no_query_utility),
        "final_prune_conflict_threshold": float(final_prune_conflict_threshold),
        "final_prune_min_keep_count": int(final_prune_min_keep),
        "final_prune_query_match_risk_reference": bool(cfg.final_prune_query_match_risk_reference),
        "final_prune_nonreference_query_match_competition": bool(
            cfg.final_prune_nonreference_query_match_competition
        ),
        "final_pruned_no_query_utility_count": int(final_pruned_no_query_utility_count),
        "final_pruned_conflict_count": int(final_pruned_conflict_count),
        "final_pruned_query_match_risk_reference_count": int(
            final_pruned_query_match_risk_reference_count
        ),
        "final_pruned_nonreference_query_match_competition_count": int(
            final_pruned_nonreference_query_match_competition_count
        ),
        "precision_fill_target_count": int(precision_fill_target),
        "precision_fill_selected_count": int(precision_fill_selected_count),
        "precision_fill_low_score_rejected_count": int(precision_fill_low_score_rejected_count),
        "precision_fill_no_query_support_rejected_count": int(
            precision_fill_no_query_support_rejected_count
        ),
        "precision_fill_cluster_rejected_count": int(precision_fill_cluster_rejected_count),
        "precision_fill_min_score": float(precision_fill_min_score),
        "precision_fill_kc_weight": float(cfg.precision_fill_kc_weight),
        "precision_fill_visibility_weight": float(cfg.precision_fill_visibility_weight),
        "precision_fill_mask_weight": float(cfg.precision_fill_mask_weight),
        "precision_fill_solver_weight": float(cfg.precision_fill_solver_weight),
        "precision_fill_query_support_weight": float(cfg.precision_fill_query_support_weight),
        "precision_fill_support_match_strength_weight": float(
            cfg.precision_fill_support_match_strength_weight
        ),
        "precision_fill_ambiguity_risk_weight": float(cfg.precision_fill_ambiguity_risk_weight),
        "precision_fill_sparse_validation_risk_weight": float(
            cfg.precision_fill_sparse_validation_risk_weight
        ),
        "precision_fill_match_competition_risk_weight": float(
            cfg.precision_fill_match_competition_risk_weight
        ),
        "precision_fill_descriptor_conflict_weight": float(
            cfg.precision_fill_descriptor_conflict_weight
        ),
        "precision_fill_require_query_support": bool(cfg.precision_fill_require_query_support),
        "precision_fill_max_per_spatial_cluster": int(precision_fill_max_per_cluster),
        "validation_min_query_landmarks": int(cfg.validation_min_query_landmarks),
        "validation_min_query_cells": int(cfg.validation_min_query_cells),
        "validation_min_query_depth_bins": int(cfg.validation_min_query_depth_bins),
        "validation_min_query_bearing_spread": float(cfg.validation_min_query_bearing_spread),
        "validation_min_query_camera_spread_m": float(cfg.validation_min_query_camera_spread_m),
        "validation_failed_query_count": int(len(validation_failed_queries)),
        "validation_failed_query_ids": sorted(validation_failed_queries),
        "validation_query_metrics": validation_query_metrics,
        "local_search_rounds": int(local_search_rounds),
        "local_search_candidates_per_query": int(local_search_candidates_per_query),
        "local_search_max_additions": int(local_search_max_additions),
        "local_search_reserve_count": int(local_search_reserve_count),
        "local_search_added_count": int(local_search_added_count),
        "local_search_failed_query_round_count": int(local_search_failed_query_round_count),
        "local_prune_conflict_threshold": float(local_prune_conflict_threshold),
        "local_prune_validate_queries": bool(cfg.local_prune_validate_queries),
        "local_pruned_conflict_count": int(local_pruned_conflict_count),
        "source_anchor_count": int(source_anchor_count),
        "source_anchor_mode": str(source_anchor_mode),
        "source_anchor_weight": float(cfg.source_anchor_weight),
        "source_anchor_soft_prior_scale": float(source_anchor_soft_prior_scale),
        "source_anchor_selected_count": int(source_anchor_selected),
        "force_source_anchor": bool(force_source_anchor),
        "augmentation_min_solver_support": float(min_aug_solver),
        "augmentation_min_query_support": float(min_aug_query),
        "augmentation_min_kc_score": float(min_aug_kc),
        "augmentation_solver_gate_rejected_count": int(augmentation_solver_gate_rejected),
        "augmentation_query_gate_rejected_count": int(augmentation_query_gate_rejected),
        "augmentation_kc_gate_rejected_count": int(augmentation_kc_gate_rejected),
        "kc_anchor_count": int(kc_anchor_count),
        "kc_anchor_min_score": float(cfg.kc_anchor_min_score),
        "kc_anchor_max_per_spatial_cluster": int(kc_anchor_max_per_cluster),
        "kc_anchor_selected_count": int(kc_anchor_selected),
        "query_prefill_target_fraction": float(cfg.query_prefill_target_fraction),
        "query_prefill_max_candidates_per_query": int(query_prefill_max_candidates),
        "query_prefill_max_per_spatial_cluster": int(query_prefill_max_per_cluster),
        "query_prefill_min_cells_per_query": int(query_prefill_min_cells),
        "query_prefill_min_depth_bins_per_query": int(query_prefill_min_depth_bins),
        "query_prefill_hard_coverage_bonus": float(query_prefill_hard_coverage_bonus),
        "query_prefill_selected_count": int(query_prefill_selected),
        "source_query_prefill_target_fraction": float(source_query_prefill_fraction),
        "source_query_prefill_max_candidates_per_query": int(source_query_prefill_max_candidates),
        "source_query_prefill_max_per_spatial_cluster": int(source_query_prefill_max_per_cluster),
        "source_query_prefill_min_cells_per_query": int(source_query_prefill_min_cells),
        "source_query_prefill_min_depth_bins_per_query": int(source_query_prefill_min_depth_bins),
        "source_query_prefill_selected_count": int(source_query_prefill_selected),
        "validation_query_prefill_target_fraction": float(validation_query_prefill_fraction),
        "validation_query_prefill_max_candidates_per_query": int(validation_query_prefill_max_candidates),
        "validation_query_prefill_max_per_spatial_cluster": int(validation_query_prefill_max_per_cluster),
        "validation_query_prefill_min_cells_per_query": int(validation_query_prefill_min_cells),
        "validation_query_prefill_min_depth_bins_per_query": int(validation_query_prefill_min_depth_bins),
        "validation_query_prefill_force_all": bool(validation_query_prefill_force_all),
        "validation_query_prefill_selected_count": int(validation_query_prefill_selected),
        "final_query_cell_counts": final_query_cell_counts,
        "final_query_depth_bin_counts": final_query_depth_bin_counts,
        "main_min_query_gain": float(cfg.main_min_query_gain),
        "main_dynamic_lookahead": int(main_dynamic_lookahead),
        "main_dynamic_effective_lookahead": int(main_dynamic_effective_lookahead),
        "main_dynamic_evaluation_count": int(main_dynamic_evaluation_count),
        "main_query_gain_gate_rejected_count": int(main_query_gain_gate_rejected),
        "main_query_match_context_gate_accepted_count": int(main_query_match_context_gate_accepted),
        "query_target_unmet_count": int(unmet_query_count),
        "selected_spatial_cluster_count": int(len(final_used_clusters)),
        "selected_depth_bin_count": int(len(final_used_depth_bins)),
        "query_geometry_weight": float(cfg.query_geometry_weight),
        "query_depth_weight": float(cfg.query_depth_weight),
        "query_bearing_weight": float(cfg.query_bearing_weight),
        "query_pnp_geometry_weight": float(cfg.query_pnp_geometry_weight),
        "query_pnp_distance_scale_m": float(cfg.query_pnp_distance_scale_m),
        "cluster_cap_rejected_count": int(cluster_cap_rejected),
        "low_gain_rejected_count": int(low_gain_rejected),
        "invalid_mask_filtered_count": int((finite & ~mask_ok).sum().item()),
        "low_visibility_filtered_count": int((finite & mask_ok & ~visibility_ok).sum().item()),
        "mean_marginal_gain": float(sum(marginal_scores) / len(marginal_scores)) if marginal_scores else 0.0,
    }
    return SparseSetSelectionResult(sampled_idx=sampled, metadata=metadata)
