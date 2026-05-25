from __future__ import annotations

import hashlib
import itertools
import math
import random
from pathlib import Path
from typing import Any

import torch

from loc_gs.diagnostics.pose_information import pose_information_summary
from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import load_feedback_bank


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


def _stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + int(digest, 16)


def _selection_set(values: torch.Tensor | Any | None) -> set[int] | None:
    if values is None:
        return None
    raw = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    return {int(item) for item in raw.tolist() if int(item) >= 0}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _xy(record: dict[str, Any]) -> torch.Tensor:
    value = record.get("keypoint_xy")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _as_float(value[0]) or 0.0
        y = _as_float(value[1]) or 0.0
        return torch.tensor([x, y], dtype=torch.float32)
    return torch.zeros(2, dtype=torch.float32)


def _record_positive(record: dict[str, Any], reprojection_threshold_px: float) -> bool:
    if bool(record.get("pnp_inlier", False)):
        return True
    error = _as_float(record.get("reprojection_error_px"))
    return error is not None and error <= float(reprojection_threshold_px)


def _quality(record: dict[str, Any], reprojection_threshold_px: float) -> float:
    score = _as_float(record.get("descriptor_score"))
    score = 1.0 if score is None else max(score, 0.0)
    error = _as_float(record.get("reprojection_error_px"))
    if error is None:
        reproj_quality = 1.0
    else:
        reproj_quality = max(0.0, 1.0 - float(error) / max(float(reprojection_threshold_px), 1e-6))
    return float(score * reproj_quality)


def _group_key(record: dict[str, Any], group_field: str) -> str:
    if str(group_field) == "query_id":
        image_id = str(record.get("image_id", "")).strip()
        if image_id:
            return image_id
    value = str(record.get(group_field, "")).strip()
    if value:
        return value.split("::", 1)[0] if str(group_field) == "query_id" else value
    query_id = str(record.get("query_id", "")).strip()
    return query_id.split("::", 1)[0] if query_id else "unknown"


def _landmark_id(record: dict[str, Any]) -> int | None:
    value = record.get("matched_gaussian_id", record.get("matched_landmark_id"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_feedback_solver_tuple_bank_for_selection(
    feedback_bank: str | Path,
    *,
    selected_idx: torch.Tensor | Any | None = None,
    xyz: torch.Tensor | Any | None = None,
    group_field: str = "image_id",
    tuple_size: int = 4,
    max_tuples_per_query: int = 512,
    top_correspondences_per_query: int = 64,
    reprojection_threshold_px: float = 8.0,
    min_all_inlier_prob: float = 0.05,
    min_logdet_h: float = -20.0,
    min_spread_2d: float = 0.0,
    min_spread_3d: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    audit = audit_feedback_bank_v2(feedback_bank)
    if audit["audit_status"] != "passed":
        raise ValueError(f"feedback bank v2 audit failed: {audit['reasons']}")
    bank = load_feedback_bank(feedback_bank)
    manifest = dict(bank.get("manifest", {}))
    selected = _selection_set(selected_idx)
    xyz_tensor: torch.Tensor | None = None
    if xyz is not None:
        xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3).cpu()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(bank.get("records", [])):
        gid = _landmark_id(record)
        if gid is None:
            continue
        if selected is not None and gid not in selected:
            continue
        if not _record_positive(record, float(reprojection_threshold_px)):
            continue
        key = _group_key(record, group_field)
        grouped.setdefault(key, []).append(
            {
                "row": int(index),
                "query_id": str(record.get("query_id", "")),
                "keypoint_id": str(record.get("keypoint_id", "")),
                "landmark_id": int(gid),
                "xy": _xy(record),
                "xyz": xyz_tensor[int(gid)] if xyz_tensor is not None and 0 <= int(gid) < xyz_tensor.shape[0] else None,
                "quality": _quality(record, float(reprojection_threshold_px)),
            }
        )

    query_reports: list[dict[str, Any]] = []
    total_support = 0
    total_weighted_support = 0.0
    total_tuple_count = 0
    total_viable_count = 0
    total_viable_mass = 0.0
    logdet_values: list[float] = []
    ambiguity_values: list[float] = []
    for query_id in sorted(grouped):
        records = sorted(
            grouped[query_id],
            key=lambda item: (-float(item["quality"]), str(item["keypoint_id"]), int(item["landmark_id"])),
        )
        if int(top_correspondences_per_query) > 0:
            records = records[: int(top_correspondences_per_query)]
        tuple_reports: list[dict[str, Any]] = []
        for tuple_id, indices in enumerate(
            _tuple_indices(
                len(records),
                int(tuple_size),
                int(max_tuples_per_query),
                _stable_seed(query_id, int(seed)),
            )
        ):
            items = [records[index] for index in indices]
            if len({str(item["keypoint_id"]) for item in items}) != len(items):
                continue
            if len({int(item["landmark_id"]) for item in items}) != len(items):
                continue
            xy_points = torch.stack([item["xy"] for item in items], dim=0)
            xyz_points = None
            if xyz_tensor is not None and all(item["xyz"] is not None for item in items):
                xyz_points = torch.stack([item["xyz"] for item in items if item["xyz"] is not None], dim=0)
            weights = torch.tensor([float(item["quality"]) for item in items], dtype=torch.float32)
            info = pose_information_summary(xy_points, xyz_points, weights=weights)
            all_inlier_prob = float(weights.clamp(0.0, 1.0).prod().item())
            ambiguity_risk = 0.0
            viable = (
                all_inlier_prob >= float(min_all_inlier_prob)
                and float(info["logdet_H"]) >= float(min_logdet_h)
                and float(info["spatial_spread_2d"]) >= float(min_spread_2d)
                and float(info["spatial_spread_3d"]) >= float(min_spread_3d)
            )
            tuple_reports.append(
                {
                    "tuple_id": int(tuple_id),
                    "landmark_ids": [int(item["landmark_id"]) for item in items],
                    "keypoint_ids": [str(item["keypoint_id"]) for item in items],
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
        mean_ambiguity = 0.0
        query_reports.append(
            {
                "query_id": str(query_id),
                "support_count": int(len(records)),
                "weighted_support": float(sum(float(item["quality"]) for item in records)),
                "tuple_count": int(tuple_count),
                "viable_tuple_count": int(viable_count),
                "viable_tuple_mass": viable_mass,
                "mean_logdet_H": mean_logdet,
                "mean_ambiguity_risk": mean_ambiguity,
                "tuples": tuple_reports[: min(tuple_count, 32)],
            }
        )
        total_support += len(records)
        total_weighted_support += sum(float(item["quality"]) for item in records)
        total_tuple_count += tuple_count
        total_viable_count += viable_count
        total_viable_mass += viable_mass
        if tuple_count:
            logdet_values.append(mean_logdet)
            ambiguity_values.append(mean_ambiguity)

    summary = {
        "query_count": int(len(query_reports)),
        "query_group_mode": f"feedback_bank_v2:{group_field}",
        "support_count_sum": int(total_support),
        "weighted_support_sum": float(total_weighted_support),
        "tuple_count": int(total_tuple_count),
        "viable_tuple_count": int(total_viable_count),
        "viable_tuple_mass": float(total_viable_mass),
        "mean_logdet_H": float(sum(logdet_values) / len(logdet_values)) if logdet_values else 0.0,
        "mean_ambiguity_risk": float(sum(ambiguity_values) / len(ambiguity_values)) if ambiguity_values else 0.0,
        "selected_count": None if selected is None else int(len(selected)),
        "tuple_size": int(tuple_size),
        "max_tuples_per_query": int(max_tuples_per_query),
        "top_correspondences_per_query": int(top_correspondences_per_query),
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "split_audit": manifest.get("split_audit", {}),
        "schema_version": manifest.get("schema_version", ""),
    }
    return {"summary": summary, "queries": query_reports}


def compare_feedback_solver_tuple_banks(
    feedback_bank: str | Path,
    *,
    source_idx: torch.Tensor | Any | None = None,
    candidate_idx: torch.Tensor | Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    source = build_feedback_solver_tuple_bank_for_selection(feedback_bank, selected_idx=source_idx, **kwargs)
    candidate = build_feedback_solver_tuple_bank_for_selection(feedback_bank, selected_idx=candidate_idx, **kwargs)
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
    source_queries = {str(item["query_id"]): item for item in source["queries"]}
    candidate_queries = {str(item["query_id"]): item for item in candidate["queries"]}
    query_deltas: list[dict[str, Any]] = []
    for query_id in sorted(set(source_queries) | set(candidate_queries)):
        src = source_queries.get(query_id, {})
        cand = candidate_queries.get(query_id, {})
        query_delta = {"query_id": str(query_id)}
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
