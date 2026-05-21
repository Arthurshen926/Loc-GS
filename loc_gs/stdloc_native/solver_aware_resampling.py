from __future__ import annotations

from typing import Any, Callable

import torch


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


def solver_aware_local_edit(
    *,
    source_idx: torch.Tensor | Any,
    candidate_pool: torch.Tensor | Any,
    utility: torch.Tensor | Any,
    evidence_mask: torch.Tensor | Any,
    safe_core: torch.Tensor | Any | None = None,
    is_admissible: Callable[[int, int], bool] | None = None,
    max_edits: int = 256,
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
    removable = [
        int(item)
        for item in selected
        if int(item) not in safe
    ]
    edits: list[dict[str, Any]] = []
    rejected_by_admissibility = 0
    rejected_low_gain = 0
    for add_id in candidates:
        if len(edits) >= int(max_edits):
            break
        if not removable:
            break
        drop_id = min(removable, key=lambda idx: (float(utility_t[idx].item()), idx))
        gain = float(utility_t[add_id].item()) - float(utility_t[drop_id].item())
        if gain <= 0.0:
            rejected_low_gain += 1
            continue
        if is_admissible is not None and not bool(is_admissible(add_id, drop_id)):
            rejected_by_admissibility += 1
            continue
        selected.remove(drop_id)
        selected.add(add_id)
        removable.remove(drop_id)
        if add_id not in safe:
            removable.append(add_id)
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
