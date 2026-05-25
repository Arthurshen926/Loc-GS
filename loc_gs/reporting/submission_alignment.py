from __future__ import annotations

from typing import Any, Mapping, Sequence


_REQUIRED_DENSE_METRICS = (
    "median_te_cm",
    "median_re_deg",
    "recall_10cm_5deg",
    "recall_5cm_5deg",
    "recall_2cm_2deg",
)
_REPORTING_FIELDS = ("runtime", "memory", "map_size", "edit_budget")
_FIELD_ALIASES = {
    "runtime": ("runtime", "runtime_ms", "latency_ms", "timing", "timing_profile", "seconds_per_query"),
    "memory": ("memory", "memory_mb", "peak_memory_mb", "peak_gpu_mb", "gpu_memory_mb"),
    "map_size": ("map_size", "map_size_mb", "landmark_count", "sampled_count"),
    "edit_budget": ("edit_budget", "max_edits", "same_budget", "native_dropped_count", "added_non_native_count"),
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dense_metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = _mapping(row.get("metrics"))
    dense = _mapping(metrics.get("dense"))
    if dense:
        return dense
    return _mapping(row.get("dense"))


def _has_reporting_field(row: Mapping[str, Any], field: str) -> bool:
    aliases = _FIELD_ALIASES[field]
    containers = [
        row,
        _mapping(row.get("reporting")),
        _mapping(row.get("metrics")),
        _mapping(row.get("manifest")),
        _mapping(row.get("metadata")),
        _mapping(_mapping(row.get("manifest")).get("solver_aware")),
        _mapping(_mapping(row.get("manifest")).get("evidence_gate")),
    ]
    for container in containers:
        if any(alias in container and container[alias] is not None for alias in aliases):
            return True
    return False


def _reference_methods(references: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not references:
        return []
    methods = references.get("methods", []) if isinstance(references, Mapping) else []
    return [dict(method) for method in methods if isinstance(method, Mapping)]


def _requires_local_reproduction(method: Mapping[str, Any]) -> bool:
    if "requires_local_reproduction" in method:
        return bool(method.get("requires_local_reproduction"))
    if method.get("reference_mode") == "paper_reported":
        return False
    if method.get("paper_reported_reference") is True:
        return False
    return True


def build_submission_alignment_report(
    board: Mapping[str, Any],
    *,
    references: Mapping[str, Any] | None = None,
    required_dense_metrics: Sequence[str] = _REQUIRED_DENSE_METRICS,
    required_reporting_fields: Sequence[str] = _REPORTING_FIELDS,
) -> dict[str, Any]:
    """Build a reporting-only readiness view for STDLoc/ULF-Loc style submission tables."""

    runs: list[dict[str, Any]] = []
    for index, raw_row in enumerate(board.get("runs", []) if isinstance(board, Mapping) else []):
        row = _mapping(raw_row)
        dense = _dense_metrics(row)
        missing_dense = [name for name in required_dense_metrics if name not in dense or dense[name] is None]
        missing_reporting = [name for name in required_reporting_fields if not _has_reporting_field(row, name)]
        paper_safe = bool(row.get("paper_safe", False))
        ready = paper_safe and not missing_dense and not missing_reporting
        runs.append(
            {
                "run_name": str(row.get("run_name", row.get("name", f"run_{index}"))),
                "scene": row.get("scene", "unknown"),
                "run_role": row.get("run_role", row.get("role", "unknown")),
                "paper_safe": paper_safe,
                "paper_safety_reason": row.get("paper_safety_reason", ""),
                "missing_dense_metrics": missing_dense,
                "missing_reporting_fields": missing_reporting,
                "ready_for_main_table": bool(ready),
            }
        )

    methods = _reference_methods(references)
    missing_reproductions = [
        str(method.get("name", "unknown"))
        for method in methods
        if _requires_local_reproduction(method) and not bool(method.get("code_reproduced", False))
    ]
    paper_reported_references = [
        str(method.get("name", "unknown"))
        for method in methods
        if not _requires_local_reproduction(method) and not bool(method.get("code_reproduced", False))
    ]
    submission_ready = bool(runs) and all(row["ready_for_main_table"] for row in runs) and not missing_reproductions
    return {
        "submission_ready": bool(submission_ready),
        "runs": runs,
        "reference_methods": methods,
        "missing_reproductions": missing_reproductions,
        "paper_reported_references": paper_reported_references,
        "required_dense_metrics": list(required_dense_metrics),
        "required_reporting_fields": list(required_reporting_fields),
    }
