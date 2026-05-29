from __future__ import annotations

from typing import Any


def build_large_edit_resampling_policy(
    *,
    source_count: int,
    replacement_fraction: float,
    saturation_enabled: bool,
    conflict_graph_available: bool,
    min_safe_core_fraction: float = 0.2,
    large_edit_threshold: float = 0.05,
    max_replacement_fraction: float = 0.30,
) -> dict[str, Any]:
    """Validate and materialize a safe bank-level large-edit resampling policy."""

    source = int(source_count)
    if source <= 0:
        raise ValueError("source_count must be positive")
    fraction = float(replacement_fraction)
    if fraction < 0.0:
        raise ValueError("replacement_fraction must be non-negative")
    if fraction > float(max_replacement_fraction):
        raise ValueError("replacement_fraction exceeds max_replacement_fraction")
    large_edit = fraction > float(large_edit_threshold)
    if large_edit and not bool(saturation_enabled):
        raise ValueError("large-edit resampling requires saturation control")
    if large_edit and not bool(conflict_graph_available):
        raise ValueError("large-edit resampling requires negative conflict graph control")
    max_edits = int(round(source * fraction))
    safe_core_min = int(round(source * max(0.0, min(1.0, float(min_safe_core_fraction)))))
    return {
        "stage": "bank_level_resampling" if large_edit else "small_edit_validation",
        "source_count": int(source),
        "replacement_fraction": float(fraction),
        "max_edits": int(max_edits),
        "safe_core_min_count": int(safe_core_min),
        "saturation_enabled": bool(saturation_enabled),
        "conflict_graph_available": bool(conflict_graph_available),
        "requires_train_dev_gate": bool(large_edit),
        "paper_safe_scope": "fixed_reconstruction_time_policy",
    }
