from __future__ import annotations

from typing import Any


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def timing_profile_digest(timing_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Extract paper-facing online timing fields from a profiler payload."""

    timing_profile = dict(timing_profile or {})
    latency = timing_profile.get("latency_ms", {})
    digest: dict[str, Any] = {}
    if isinstance(latency, dict):
        for stage, summary in latency.items():
            if not isinstance(summary, dict):
                continue
            digest[f"{stage}_ms"] = {
                key: _float_or_zero(summary.get(key))
                for key in ("mean", "median", "p95")
                if key in summary
            }
    fps = timing_profile.get("fps", {})
    if isinstance(fps, dict):
        digest["fps"] = _float_or_zero(fps.get("mean_latency"))
    else:
        digest["fps"] = 0.0
    return digest


def merge_offline_costs(
    *,
    base_map_s: float | int | None = None,
    feedback_cache_s: float | int | None = None,
    selector_train_s: float | int | None = None,
    export_s: float | int | None = None,
    peak_gpu_mb: float | int | None = None,
    map_size_mb: float | int | None = None,
) -> dict[str, float]:
    """Normalize offline costs and separate base-map cost from Loc-GS overhead."""

    base = _float_or_zero(base_map_s)
    feedback = _float_or_zero(feedback_cache_s)
    train = _float_or_zero(selector_train_s)
    export = _float_or_zero(export_s)
    extra = feedback + train + export
    return {
        "base_map_s": base,
        "feedback_cache_s": feedback,
        "selector_train_s": train,
        "export_s": export,
        "locgs_extra_s": extra,
        "total_offline_s": base + extra,
        "peak_gpu_mb": _float_or_zero(peak_gpu_mb),
        "map_size_mb": _float_or_zero(map_size_mb),
    }

