from __future__ import annotations

import math
from typing import Any, Mapping

import torch


def _index_set(values: torch.Tensor | Any | None) -> set[int]:
    if values is None:
        return set()
    tensor = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    return {int(value) for value in tensor.tolist() if int(value) >= 0}


def _float_vector(values: torch.Tensor | Any) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()


def _support_table(
    raw: object,
    *,
    prefix: str,
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    if not isinstance(raw, Mapping):
        return out
    for raw_query_id, raw_landmarks in raw.items():
        if not isinstance(raw_landmarks, Mapping):
            continue
        query_id = f"{prefix}:{raw_query_id}"
        for raw_gid, raw_value in raw_landmarks.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if gid < 0 or not math.isfinite(value) or value <= 0.0:
                continue
            out.setdefault(query_id, {})
            out[query_id][gid] = float(out[query_id].get(gid, 0.0) + value)
    return out


def _flatten_support(table: Mapping[str, Mapping[int, float]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for landmarks in table.values():
        for gid, value in landmarks.items():
            out[int(gid)] = float(out.get(int(gid), 0.0) + max(0.0, float(value)))
    return out


def _support_retention(
    table: Mapping[str, Mapping[int, float]],
    selected_set: set[int],
) -> dict[str, Any]:
    flat = _flatten_support(table)
    selected_ids = {gid for gid in flat if gid in selected_set}
    total_mass = float(sum(max(0.0, float(value)) for value in flat.values()))
    selected_mass = float(sum(max(0.0, float(flat[gid])) for gid in selected_ids))
    return {
        "query_count": int(len(table)),
        "entry_count": int(sum(len(values) for values in table.values())),
        "landmark_count": int(len(flat)),
        "landmark_selected_count": int(len(selected_ids)),
        "landmark_missing_count": int(len(set(flat) - selected_set)),
        "landmark_selected_fraction": float(len(selected_ids) / len(flat)) if flat else 0.0,
        "support_mass": total_mass,
        "support_mass_selected": selected_mass,
        "support_mass_selected_fraction": float(selected_mass / total_mass) if total_mass > 0.0 else 0.0,
    }


def _risk_stats(raw_risk: object, selected_set: set[int]) -> dict[str, Any]:
    risk: dict[int, float] = {}
    if isinstance(raw_risk, Mapping):
        for raw_gid, raw_value in raw_risk.items():
            try:
                gid = int(raw_gid)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if gid >= 0 and math.isfinite(value) and value > 0.0:
                risk[gid] = max(float(risk.get(gid, 0.0)), float(value))
    selected_ids = {gid for gid in risk if gid in selected_set}
    total_mass = float(sum(risk.values()))
    selected_mass = float(sum(risk[gid] for gid in selected_ids))
    return {
        "count": int(len(risk)),
        "selected_count": int(len(selected_ids)),
        "selected_fraction": float(len(selected_ids) / len(risk)) if risk else 0.0,
        "mass": total_mass,
        "selected_mass": selected_mass,
        "selected_mass_fraction": float(selected_mass / total_mass) if total_mass > 0.0 else 0.0,
    }


def _per_query_support_retention(
    protected: Mapping[str, Mapping[int, float]],
    validated: Mapping[str, Mapping[int, float]],
    selected_set: set[int],
) -> dict[str, dict[str, float]]:
    merged: dict[str, Mapping[int, float]] = {}
    merged.update(protected)
    merged.update(validated)
    out: dict[str, dict[str, float]] = {}
    for query_id, landmarks in merged.items():
        total = float(sum(max(0.0, float(value)) for value in landmarks.values()))
        selected = float(
            sum(max(0.0, float(value)) for gid, value in landmarks.items() if int(gid) in selected_set)
        )
        total_count = int(len(landmarks))
        selected_count = int(sum(1 for gid in landmarks if int(gid) in selected_set))
        out[str(query_id)] = {
            "landmark_count": float(total_count),
            "selected_landmark_count": float(selected_count),
            "selected_landmark_fraction": float(selected_count / total_count) if total_count > 0 else 0.0,
            "support_sum": total,
            "selected_support_sum": selected,
            "selected_support_fraction": float(selected / total) if total > 0.0 else 0.0,
        }
    return out


def _mean_for_ids(values: torch.Tensor, ids: set[int]) -> float | None:
    valid = [gid for gid in ids if 0 <= int(gid) < int(values.numel())]
    if not valid:
        return None
    tensor = values[torch.tensor(sorted(valid), dtype=torch.long)]
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) == 0:
        return None
    return float(finite.mean().item())


def _score_stats(
    score_vectors: Mapping[str, torch.Tensor | Any],
    *,
    selected_set: set[int],
    baseline_set: set[int],
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    missing_baseline = baseline_set - selected_set
    selected_baseline = baseline_set & selected_set
    for name, raw_values in score_vectors.items():
        values = _float_vector(raw_values)
        out[str(name)] = {
            "selected_mean": _mean_for_ids(values, selected_set),
            "baseline_mean": _mean_for_ids(values, baseline_set),
            "selected_baseline_mean": _mean_for_ids(values, selected_baseline),
            "missing_baseline_mean": _mean_for_ids(values, missing_baseline),
        }
    return out


def audit_sparse_set_composition(
    *,
    selected_idx: torch.Tensor | Any,
    baseline_idx: torch.Tensor | Any | None = None,
    sparse_validation_profile: Mapping[str, Any] | None = None,
    score_vectors: Mapping[str, torch.Tensor | Any] | None = None,
) -> dict[str, Any]:
    """Audit what a SparseSet resample kept, lost, and risked.

    This function is intentionally independent of Cambridge/ULF runtime code so
    it can be used both in unit tests and post-hoc artifact reports.
    """

    selected_set = _index_set(selected_idx)
    baseline_set = _index_set(baseline_idx)
    profile = sparse_validation_profile or {}
    protected = _support_table(profile.get("protected_per_query_support", {}), prefix="protected")
    validated = _support_table(profile.get("validated_per_query_support", {}), prefix="validated")
    overlap = selected_set & baseline_set
    return {
        "selected_count": int(len(selected_set)),
        "baseline_count": int(len(baseline_set)),
        "baseline_overlap_count": int(len(overlap)),
        "baseline_overlap_fraction": float(len(overlap) / len(baseline_set)) if baseline_set else 0.0,
        "protected_support": _support_retention(protected, selected_set),
        "validated_support": _support_retention(validated, selected_set),
        "regression_risk": _risk_stats(profile.get("landmark_regression_risk", {}), selected_set),
        "per_query_support": _per_query_support_retention(protected, validated, selected_set),
        "score_stats": _score_stats(score_vectors or {}, selected_set=selected_set, baseline_set=baseline_set),
    }

