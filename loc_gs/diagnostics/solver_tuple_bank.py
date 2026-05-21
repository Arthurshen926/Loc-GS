from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Any

import torch

from loc_gs.diagnostics.ambiguity_metrics import descriptor_ambiguity_risk
from loc_gs.diagnostics.pose_information import pose_information_summary


def _load_payload(payload_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload_or_path, (str, Path)):
        return torch.load(Path(payload_or_path), map_location="cpu")
    return dict(payload_or_path)


def _cache_tensor(payload: dict[str, Any], *names: str) -> torch.Tensor | None:
    for name in names:
        if name in payload:
            return torch.as_tensor(payload[name])
    return None


def _positive_mask(
    payload: dict[str, Any],
    ids: torch.Tensor,
    valid: torch.Tensor,
    *,
    score_threshold: float | None,
    reprojection_threshold_px: float | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    metadata = dict(payload.get("metadata", {}))
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        valid = valid & finite
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        quality = (1.0 - error_tensor / float(reprojection_threshold_px)).clamp(0.0, 1.0)
    elif pair_label is not None:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
        quality = torch.ones_like(score_tensor)
    else:
        raise KeyError("episode cache needs reprojection_error or pair_label")
    if score_threshold is not None:
        positive = positive & (score_tensor >= float(score_threshold))
    return valid & positive, quality * score_tensor.clamp_min(0.0), float(reprojection_threshold_px)


def _mapped_ids(ids: torch.Tensor, base_gaussian_id: torch.Tensor | Any | None, num_gaussians: int) -> torch.Tensor:
    if base_gaussian_id is None:
        mapped = ids.long()
    else:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped = base_ids[ids.clamp_min(0).long()]
    if mapped.numel() and (int(mapped.min().item()) < 0 or int(mapped.max().item()) >= int(num_gaussians)):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    return mapped


def _query_ids(payload: dict[str, Any], row_count: int, synthetic_group_size: int | None) -> tuple[torch.Tensor, str]:
    query = _cache_tensor(payload, "query_id")
    if query is not None:
        query = query.long().reshape(-1).cpu()
        if query.shape[0] != int(row_count):
            raise ValueError("query_id must have one value per candidate row")
        return query, "query_id"
    if synthetic_group_size is not None and int(synthetic_group_size) > 0:
        group_size = int(synthetic_group_size)
        return torch.arange(row_count, dtype=torch.long) // group_size, f"synthetic_chunks_{group_size}"
    return torch.arange(row_count, dtype=torch.long), "row"


def _tuple_indices(count: int, tuple_size: int, max_tuples: int, seed: int) -> list[tuple[int, ...]]:
    if count < tuple_size:
        return []
    total = math.comb(count, tuple_size)
    if total <= int(max_tuples):
        return list(itertools.combinations(range(count), tuple_size))
    rng = random.Random(int(seed))
    tuples: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(int(max_tuples) * 20, 100)
    while len(tuples) < int(max_tuples) and attempts < max_attempts:
        attempts += 1
        tuples.add(tuple(sorted(rng.sample(range(count), tuple_size))))
    return sorted(tuples)


def _selection_set(values: torch.Tensor | Any, num_gaussians: int) -> set[int]:
    raw = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    return {int(item) for item in raw.tolist() if 0 <= int(item) < int(num_gaussians)}


def build_solver_tuple_bank_for_selection(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    selected_idx: torch.Tensor | Any,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    xyz: torch.Tensor | Any | None = None,
    tuple_size: int = 4,
    max_tuples_per_query: int = 512,
    top_correspondences_per_query: int = 64,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
    synthetic_group_size: int | None = None,
    min_all_inlier_prob: float = 0.05,
    min_logdet_h: float = -20.0,
    min_spread_2d: float = 0.0,
    min_spread_3d: float = 0.0,
    max_ambiguity_risk: float = 1.0,
    ambiguity_cosine_threshold: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a query-grouped PnP minimal-set surrogate bank for one sampled set."""

    payload = _load_payload(payload_or_path)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long().cpu()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool().cpu()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    positive, quality, reproj_threshold = _positive_mask(
        payload,
        ids,
        valid,
        score_threshold=score_threshold,
        reprojection_threshold_px=reprojection_threshold_px,
    )
    mapped = _mapped_ids(ids, base_gaussian_id, int(num_gaussians))
    selected = _selection_set(selected_idx, int(num_gaussians))
    xy = _cache_tensor(payload, "query_yx", "keypoint_yx")
    if xy is None:
        xy_tensor = torch.zeros((ids.shape[0], 2), dtype=torch.float32)
    else:
        xy_tensor = xy.float().reshape(-1, 2).cpu()
        if xy_tensor.shape[0] != ids.shape[0]:
            raise ValueError("query_yx/keypoint_yx must have one row per candidate row")
    xyz_tensor: torch.Tensor | None = None
    if xyz is not None:
        xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3).cpu()
        if xyz_tensor.shape[0] != int(num_gaussians):
            raise ValueError("xyz must have one row per Gaussian")
    desc = _cache_tensor(payload, "base_landmark_desc")
    if desc is not None:
        desc = desc.float().cpu()
    query_id, query_group_mode = _query_ids(payload, int(ids.shape[0]), synthetic_group_size)
    selected_mask = torch.zeros_like(mapped, dtype=torch.bool)
    if selected:
        selected_mask = torch.tensor(
            [[int(gid) in selected for gid in row.tolist()] for row in mapped],
            dtype=torch.bool,
        )
    usable = valid & positive & selected_mask
    query_reports: list[dict[str, Any]] = []
    total_support = 0
    total_weighted_support = 0.0
    total_tuple_count = 0
    total_viable_count = 0
    total_viable_mass = 0.0
    logdet_values: list[float] = []
    ambiguity_values: list[float] = []
    for q_value in torch.unique(query_id, sorted=True).tolist():
        rows = torch.where(query_id == int(q_value))[0]
        records: list[dict[str, Any]] = []
        for row in rows.tolist():
            cols = torch.where(usable[row])[0].tolist()
            for col in cols:
                gid = int(mapped[row, col].item())
                record = {
                    "row": int(row),
                    "landmark_id": gid,
                    "xy": xy_tensor[row],
                    "xyz": xyz_tensor[gid] if xyz_tensor is not None else None,
                    "quality": float(quality[row, col].item()),
                    "descriptor": desc[int(ids[row, col].item())] if desc is not None and int(ids[row, col].item()) < desc.shape[0] else None,
                }
                records.append(record)
        records = sorted(records, key=lambda item: (-float(item["quality"]), int(item["row"]), int(item["landmark_id"])))
        if int(top_correspondences_per_query) > 0:
            records = records[: int(top_correspondences_per_query)]
        support_count = len(records)
        weighted_support = sum(float(item["quality"]) for item in records)
        tuple_reports: list[dict[str, Any]] = []
        for tuple_id, indices in enumerate(
            _tuple_indices(
                len(records),
                int(tuple_size),
                int(max_tuples_per_query),
                int(seed) + int(q_value),
            )
        ):
            items = [records[index] for index in indices]
            if len({int(item["row"]) for item in items}) != len(items):
                continue
            if len({int(item["landmark_id"]) for item in items}) != len(items):
                continue
            xy_points = torch.stack([item["xy"] for item in items], dim=0)
            xyz_points = None
            if xyz_tensor is not None:
                xyz_points = torch.stack([item["xyz"] for item in items], dim=0)
            weights = torch.tensor([float(item["quality"]) for item in items], dtype=torch.float32)
            info = pose_information_summary(xy_points, xyz_points, weights=weights)
            desc_items = [item["descriptor"] for item in items]
            if all(item is not None for item in desc_items):
                ambiguity = descriptor_ambiguity_risk(
                    torch.stack([item for item in desc_items if item is not None], dim=0),
                    cosine_threshold=float(ambiguity_cosine_threshold),
                )
                ambiguity_risk = float(ambiguity["ambiguity_risk"])
            else:
                ambiguity_risk = 0.0
            all_inlier_prob = float(weights.clamp(0.0, 1.0).prod().item())
            viable = (
                all_inlier_prob >= float(min_all_inlier_prob)
                and float(info["logdet_H"]) >= float(min_logdet_h)
                and float(info["spatial_spread_2d"]) >= float(min_spread_2d)
                and float(info["spatial_spread_3d"]) >= float(min_spread_3d)
                and ambiguity_risk <= float(max_ambiguity_risk)
            )
            tuple_reports.append(
                {
                    "tuple_id": int(tuple_id),
                    "landmark_ids": [int(item["landmark_id"]) for item in items],
                    "all_inlier_prob": all_inlier_prob,
                    "geometry_logdet": float(info["logdet_H"]),
                    "geometry_condition": float(info["condition_number_H"]),
                    "spatial_spread_2d": float(info["spatial_spread_2d"]),
                    "spatial_spread_3d": float(info["spatial_spread_3d"]),
                    "ambiguity_risk": ambiguity_risk,
                    "viable": bool(viable),
                }
            )
        tuple_count = len(tuple_reports)
        viable_tuples = [item for item in tuple_reports if item["viable"]]
        viable_count = len(viable_tuples)
        viable_mass = float(sum(float(item["all_inlier_prob"]) for item in viable_tuples))
        mean_logdet = float(sum(float(item["geometry_logdet"]) for item in tuple_reports) / tuple_count) if tuple_count else 0.0
        mean_ambiguity = (
            float(sum(float(item["ambiguity_risk"]) for item in tuple_reports) / tuple_count) if tuple_count else 0.0
        )
        query_reports.append(
            {
                "query_id": int(q_value),
                "support_count": int(support_count),
                "weighted_support": float(weighted_support),
                "tuple_count": int(tuple_count),
                "viable_tuple_count": int(viable_count),
                "viable_tuple_mass": viable_mass,
                "mean_logdet_H": mean_logdet,
                "mean_ambiguity_risk": mean_ambiguity,
                "tuples": tuple_reports[: min(tuple_count, 32)],
            }
        )
        total_support += support_count
        total_weighted_support += weighted_support
        total_tuple_count += tuple_count
        total_viable_count += viable_count
        total_viable_mass += viable_mass
        if tuple_count:
            logdet_values.append(mean_logdet)
            ambiguity_values.append(mean_ambiguity)
    metadata = dict(payload.get("metadata", {}))
    summary = {
        "query_count": int(len(query_reports)),
        "query_group_mode": query_group_mode,
        "support_count_sum": int(total_support),
        "weighted_support_sum": float(total_weighted_support),
        "tuple_count": int(total_tuple_count),
        "viable_tuple_count": int(total_viable_count),
        "viable_tuple_mass": float(total_viable_mass),
        "mean_logdet_H": float(sum(logdet_values) / len(logdet_values)) if logdet_values else 0.0,
        "mean_ambiguity_risk": float(sum(ambiguity_values) / len(ambiguity_values)) if ambiguity_values else 0.0,
        "selected_count": int(len(selected)),
        "tuple_size": int(tuple_size),
        "max_tuples_per_query": int(max_tuples_per_query),
        "top_correspondences_per_query": int(top_correspondences_per_query),
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reproj_threshold),
        "split_audit": metadata.get("split_audit", {}),
    }
    return {
        "summary": summary,
        "queries": query_reports,
    }


def compare_selection_solver_diagnostics(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    source_idx: torch.Tensor | Any,
    candidate_idx: torch.Tensor | Any,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    xyz: torch.Tensor | Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compare source and candidate sampled sets with solver-level diagnostics."""

    source = build_solver_tuple_bank_for_selection(
        payload_or_path,
        selected_idx=source_idx,
        num_gaussians=int(num_gaussians),
        base_gaussian_id=base_gaussian_id,
        xyz=xyz,
        **kwargs,
    )
    candidate = build_solver_tuple_bank_for_selection(
        payload_or_path,
        selected_idx=candidate_idx,
        num_gaussians=int(num_gaussians),
        base_gaussian_id=base_gaussian_id,
        xyz=xyz,
        **kwargs,
    )
    scalar_keys = (
        "support_count_sum",
        "weighted_support_sum",
        "tuple_count",
        "viable_tuple_count",
        "viable_tuple_mass",
        "mean_logdet_H",
        "mean_ambiguity_risk",
    )
    delta = {
        key: candidate["summary"][key] - source["summary"][key]
        for key in scalar_keys
        if key in source["summary"] and key in candidate["summary"]
    }
    source_queries = {int(item["query_id"]): item for item in source["queries"]}
    candidate_queries = {int(item["query_id"]): item for item in candidate["queries"]}
    query_deltas: list[dict[str, Any]] = []
    for query_id in sorted(set(source_queries) | set(candidate_queries)):
        src = source_queries.get(query_id, {})
        cand = candidate_queries.get(query_id, {})
        query_delta = {"query_id": int(query_id)}
        for key in (
            "support_count",
            "weighted_support",
            "tuple_count",
            "viable_tuple_count",
            "viable_tuple_mass",
            "mean_logdet_H",
            "mean_ambiguity_risk",
        ):
            query_delta[f"{key}_delta"] = float(cand.get(key, 0.0)) - float(src.get(key, 0.0))
        query_deltas.append(query_delta)
    return {
        "source": source,
        "candidate": candidate,
        "summary_delta": delta,
        "query_deltas": query_deltas,
    }
