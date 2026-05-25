from __future__ import annotations

import math
import bisect
from typing import Any, Callable, Mapping

import torch

_QUERY_DELTA_FIELDS = (
    "support",
    "viable_tuple_mass",
    "logdet_H",
    "min_eigenvalue",
    "dense_worsen_risk",
    "ambiguity",
)
_QUERY_DELTA_FIELD_ALIASES = {
    "min_eigen": "min_eigenvalue",
    "min_eigen_H": "min_eigenvalue",
    "lambda_min": "min_eigenvalue",
    "dense_worsen": "dense_worsen_risk",
    "dense_risk": "dense_worsen_risk",
}
_DEFAULT_QUERY_UTILITY_WEIGHTS = {
    "support": 1.0,
    "viable_tuple_mass": 1.0,
    "logdet_H": 1.0,
    "min_eigenvalue": 1.0,
    "dense_worsen_risk": -1.0,
    "ambiguity": -1.0,
}


def parse_edit_selection(selection: str, *, total_edits: int) -> list[int]:
    """Parse a one-based edit selection string into zero-based edit indices."""

    total = int(total_edits)
    if total < 0:
        raise ValueError("total_edits must be non-negative")
    text = str(selection or "").strip().lower()
    if not text:
        raise ValueError("selection must not be empty")
    if text == "all":
        return list(range(total))
    selected: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.startswith("prefix:"):
            count = int(part.split(":", 1)[1])
            if count < 0:
                raise ValueError("prefix count must be non-negative")
            selected.update(range(min(count, total)))
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo = int(lo_text)
            hi = int(hi_text)
            if lo <= 0 or hi <= 0 or hi < lo:
                raise ValueError(f"invalid edit range: {raw_part}")
            indices = range(lo - 1, hi)
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"invalid edit index: {raw_part}")
            indices = range(value - 1, value)
        for index in indices:
            if index < 0 or index >= total:
                raise ValueError(f"edit index {index + 1} is outside available edits 1..{total}")
            selected.add(int(index))
    return sorted(selected)


def _long_vector(values: torch.Tensor | Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    if tensor.numel() == 0 and name != "safe_core":
        raise ValueError(f"{name} must not be empty")
    return tensor


def _float_vector(values: torch.Tensor | Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor


def _finite_float(value: Any, *, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def score_hard_query_cvar_utility(
    query_deltas: Mapping[str | int, Mapping[str, Any]],
    *,
    weights: Mapping[str, float] | None = None,
    alpha: float = 0.2,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Score query-level solver deltas with mean utility plus lower-tail CVaR."""

    if not isinstance(query_deltas, Mapping):
        raise TypeError("query_deltas must be a mapping keyed by query id")
    if not query_deltas:
        raise ValueError("query_deltas must not be empty")
    alpha_f = _finite_float(alpha, name="alpha")
    if alpha_f <= 0.0 or alpha_f > 1.0:
        raise ValueError("alpha must be in the interval (0, 1]")
    min_score_f = _finite_float(min_score, name="min_score")
    metric_weights = dict(_DEFAULT_QUERY_UTILITY_WEIGHTS)
    if weights is not None:
        for name, value in weights.items():
            canonical = _QUERY_DELTA_FIELD_ALIASES.get(str(name), str(name))
            if canonical not in metric_weights:
                raise ValueError(f"unknown utility weight: {name}")
            metric_weights[canonical] = _finite_float(value, name=f"weights[{name}]")

    per_query_utility: dict[str | int, float] = {}
    utilities: list[float] = []
    for query_id, deltas in query_deltas.items():
        if isinstance(query_id, bool) or not isinstance(query_id, (str, int)):
            raise TypeError("query ids must be strings or integers")
        if not isinstance(deltas, Mapping):
            raise TypeError(f"query_deltas[{query_id!r}] must be a mapping")
        utility = 0.0
        for field in _QUERY_DELTA_FIELDS:
            raw_delta = deltas.get(field, 0.0)
            if field not in deltas:
                for alias, canonical in _QUERY_DELTA_FIELD_ALIASES.items():
                    if canonical == field and alias in deltas:
                        raw_delta = deltas[alias]
                        break
            delta = _finite_float(raw_delta, name=f"query_deltas[{query_id!r}][{field}]")
            utility += metric_weights[field] * delta
        per_query_utility[query_id] = float(utility)
        utilities.append(float(utility))

    mean = float(sum(utilities) / len(utilities))
    tail_count = max(1, int(math.ceil(alpha_f * len(utilities))))
    tail_count = min(tail_count, len(utilities))
    tail = sorted(utilities)[:tail_count]
    cvar = float(sum(tail) / len(tail))
    score = float(mean + cvar)
    metadata = {
        "mean": mean,
        "cvar": cvar,
        "score": score,
        "alpha": alpha_f,
        "tail_count": int(tail_count),
        "query_count": int(len(utilities)),
        "min_score": min_score_f,
        "accepted": bool(score >= min_score_f),
        "weights": {name: float(value) for name, value in metric_weights.items()},
    }
    return {
        "score": score,
        "per_query_utility": per_query_utility,
        "metadata": metadata,
    }


def solver_aware_local_edit(
    *,
    source_idx: torch.Tensor | Any,
    candidate_pool: torch.Tensor | Any,
    utility: torch.Tensor | Any,
    evidence_mask: torch.Tensor | Any,
    safe_core: torch.Tensor | Any | None = None,
    is_admissible: Callable[[int, int], bool] | None = None,
    max_edits: int = 256,
    max_drop_scan: int = 1,
) -> dict[str, Any]:
    """Same-budget local replacement starting from native sampled landmarks."""

    source = _long_vector(source_idx, name="source_idx")
    pool = _long_vector(candidate_pool, name="candidate_pool")
    utility_t = _float_vector(utility, name="utility")
    evidence = torch.as_tensor(evidence_mask, dtype=torch.bool).reshape(-1).cpu()
    if evidence.shape[0] != utility_t.shape[0]:
        raise ValueError("evidence_mask and utility must have the same length")
    safe = set(_long_vector(safe_core, name="safe_core").tolist()) if safe_core is not None else set()
    selected: set[int] = {int(item) for item in source.tolist() if 0 <= int(item) < utility_t.numel()}
    source_set = set(selected)
    candidates = [
        int(item)
        for item in pool.tolist()
        if 0 <= int(item) < utility_t.numel() and int(item) not in selected and bool(evidence[int(item)].item())
    ]
    candidates = sorted(candidates, key=lambda idx: (-float(utility_t[idx].item()), idx))
    removable = sorted(
        (float(utility_t[int(item)].item()), int(item))
        for item in selected
        if int(item) not in safe
    )
    edits: list[dict[str, Any]] = []
    rejected_by_admissibility = 0
    rejected_low_gain = 0
    drop_scan_attempts = 0
    drop_scan_limit = max(1, int(max_drop_scan))
    for add_id in candidates:
        if len(edits) >= int(max_edits):
            break
        if not removable:
            break
        accepted_drop_index: int | None = None
        accepted_drop_id: int | None = None
        accepted_gain = 0.0
        for drop_index, (_, drop_value) in enumerate(removable[:drop_scan_limit]):
            drop_id = int(drop_value)
            gain = float(utility_t[add_id].item()) - float(utility_t[drop_id].item())
            drop_scan_attempts += 1
            if gain <= 0.0:
                if drop_index == 0:
                    rejected_low_gain += 1
                break
            if is_admissible is not None and not bool(is_admissible(add_id, drop_id)):
                rejected_by_admissibility += 1
                continue
            accepted_drop_index = int(drop_index)
            accepted_drop_id = int(drop_id)
            accepted_gain = float(gain)
            break
        if accepted_drop_id is None or accepted_drop_index is None:
            continue
        drop_id = accepted_drop_id
        gain = accepted_gain
        selected.remove(drop_id)
        selected.add(add_id)
        removable.pop(accepted_drop_index)
        if add_id not in safe:
            bisect.insort(removable, (float(utility_t[int(add_id)].item()), int(add_id)))
        edits.append({"add_id": int(add_id), "drop_id": int(drop_id), "utility_gain": gain})
    source_rank = {int(gid): rank for rank, gid in enumerate(source.tolist())}
    ordered = sorted(
        selected,
        key=lambda gid: (source_rank.get(int(gid), len(source_rank)), -float(utility_t[int(gid)].item()), int(gid)),
    )
    sampled = torch.tensor(ordered, dtype=torch.long)
    safe_core_dropped = len(safe & source_set - set(sampled.tolist()))
    metadata = {
        "source_sampled_count": int(source.numel()),
        "output_sampled_count": int(sampled.numel()),
        "candidate_pool_count": int(pool.numel()),
        "safe_core_count": int(len(safe)),
        "safe_core_dropped_count": int(safe_core_dropped),
        "native_kept_count": int(len(source_set & set(sampled.tolist()))),
        "native_dropped_count": int(len(source_set - set(sampled.tolist()))),
        "added_non_native_count": int(len(set(sampled.tolist()) - source_set)),
        "num_admissible_replacements": int(len(edits)),
        "num_rejected_by_admissibility": int(rejected_by_admissibility),
        "num_rejected_by_low_gain": int(rejected_low_gain),
        "drop_scan_attempts": int(drop_scan_attempts),
        "max_drop_scan": int(drop_scan_limit),
        "max_edits": int(max_edits),
    }
    return {
        "sampled_idx": sampled,
        "edits": edits,
        "metadata": metadata,
    }


def apply_selected_local_edits(
    *,
    source_idx: torch.Tensor | Any,
    edits: list[dict[str, Any]],
    utility: torch.Tensor | Any,
    selected_edit_indices: list[int] | tuple[int, ...] | torch.Tensor | Any,
    strict_conflicts: bool = False,
) -> dict[str, Any]:
    """Apply an arbitrary subset of previously generated same-budget local edits."""

    source = _long_vector(source_idx, name="source_idx")
    utility_t = _float_vector(utility, name="utility")
    selected_indices = [int(item) for item in list(selected_edit_indices)]
    selected: set[int] = {int(item) for item in source.tolist() if 0 <= int(item) < utility_t.numel()}
    source_set = set(selected)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for edit_index in selected_indices:
        if edit_index < 0 or edit_index >= len(edits):
            raise ValueError(f"edit index {edit_index} is outside available edits 0..{len(edits) - 1}")
        edit = dict(edits[edit_index])
        add_id = int(edit["add_id"])
        drop_id = int(edit["drop_id"])
        if drop_id not in selected:
            message = f"edit {edit_index + 1} cannot drop {drop_id}; it is not currently selected"
            if strict_conflicts:
                raise ValueError(message)
            skipped.append({"edit_index": int(edit_index), "add_id": add_id, "drop_id": drop_id, "reason": "drop_missing"})
            continue
        if add_id in selected:
            message = f"edit {edit_index + 1} cannot add {add_id}; it is already selected"
            if strict_conflicts:
                raise ValueError(message)
            skipped.append({"edit_index": int(edit_index), "add_id": add_id, "drop_id": drop_id, "reason": "add_present"})
            continue
        if add_id < 0 or add_id >= utility_t.numel() or drop_id < 0 or drop_id >= utility_t.numel():
            raise IndexError("edit landmark id is outside utility length")
        selected.remove(drop_id)
        selected.add(add_id)
        applied_edit = {
            "edit_index": int(edit_index),
            "add_id": add_id,
            "drop_id": drop_id,
            "utility_gain": float(edit.get("utility_gain", float(utility_t[add_id].item()) - float(utility_t[drop_id].item()))),
        }
        applied.append(applied_edit)
    source_rank = {int(gid): rank for rank, gid in enumerate(source.tolist())}
    ordered = sorted(
        selected,
        key=lambda gid: (source_rank.get(int(gid), len(source_rank)), -float(utility_t[int(gid)].item()), int(gid)),
    )
    sampled = torch.tensor(ordered, dtype=torch.long)
    metadata = {
        "source_sampled_count": int(source.numel()),
        "output_sampled_count": int(sampled.numel()),
        "same_budget": int(source.numel()) == int(sampled.numel()),
        "available_edit_count": int(len(edits)),
        "requested_edit_count": int(len(selected_indices)),
        "applied_edit_count": int(len(applied)),
        "skipped_conflict_count": int(len(skipped)),
        "native_kept_count": int(len(source_set & set(sampled.tolist()))),
        "native_dropped_count": int(len(source_set - set(sampled.tolist()))),
        "added_non_native_count": int(len(set(sampled.tolist()) - source_set)),
    }
    return {
        "sampled_idx": sampled,
        "edits": applied,
        "skipped_edits": skipped,
        "metadata": metadata,
    }
