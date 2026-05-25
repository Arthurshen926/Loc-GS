from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _map_key(scene: str, map_path: str) -> tuple[str, str]:
    return scene.strip(), map_path.strip()


def _selected_maps(frozen_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    splits = frozen_manifest.get("splits", {})
    if not isinstance(splits, Mapping):
        return []
    for split_label, split_payload in splits.items():
        if not isinstance(split_payload, Mapping):
            continue
        selected_runs = split_payload.get("selected_runs", {})
        if not isinstance(selected_runs, Mapping):
            continue
        for scene_name, raw_row in selected_runs.items():
            if not isinstance(raw_row, Mapping):
                continue
            scene = str(raw_row.get("scene", scene_name)).strip()
            map_path = str(raw_row.get("map_path", "")).strip()
            if not scene or not map_path:
                continue
            key = _map_key(scene, map_path)
            entry = by_key.setdefault(
                key,
                {
                    "scene": scene,
                    "map_path": map_path,
                    "recipe_splits": [],
                    "selected_by_split": {},
                },
            )
            entry["recipe_splits"].append(str(split_label))
            entry["selected_by_split"][str(split_label)] = raw_row.get("selected", "unknown")
    selected = []
    for entry in by_key.values():
        entry["recipe_splits"] = sorted(set(entry["recipe_splits"]))
        selected.append(entry)
    selected.sort(key=lambda item: (item["scene"], item["map_path"]))
    return selected


def _discover_profile_dirs(profile_roots: Sequence[str | Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for raw_root in profile_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        candidates = [root] if (root / "timing_profile.json").exists() else []
        candidates.extend(path.parent for path in sorted(root.rglob("timing_profile.json")) if path.parent != root)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dirs.append(candidate)
    return run_dirs


def _first_number(*values: Any) -> float | int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _profile_row(run_dir: Path) -> dict[str, Any] | None:
    manifest = _load_json(run_dir / "manifest.json")
    metrics = _load_json(run_dir / "metrics_summary.json")
    timing = _load_json(run_dir / "timing_profile.json")
    scene = str(manifest.get("scene", metrics.get("scene", timing.get("scene", "")))).strip()
    map_path = str(manifest.get("map_path", metrics.get("model_path", ""))).strip()
    if not scene or not map_path:
        return None
    memory = timing.get("memory", {}) if isinstance(timing.get("memory"), Mapping) else {}
    summary_memory = metrics.get("memory", {}) if isinstance(metrics.get("memory"), Mapping) else {}
    peak_gpu_mb = _first_number(
        timing.get("peak_gpu_mb"),
        memory.get("peak_gpu_mb"),
        metrics.get("peak_gpu_mb"),
        summary_memory.get("peak_gpu_mb"),
    )
    split = str(manifest.get("split", metrics.get("eval_split", timing.get("eval_split", "")))).strip()
    queries = _first_number(
        timing.get("queries"),
        metrics.get("max_test_cameras"),
        (manifest.get("hyperparameters", {}) if isinstance(manifest.get("hyperparameters"), Mapping) else {}).get(
            "max_test_cameras"
        ),
    )
    latency = timing.get("latency_ms", {}) if isinstance(timing.get("latency_ms"), Mapping) else {}
    total_latency = latency.get("total", {}) if isinstance(latency.get("total"), Mapping) else {}
    split_audit = _load_json(run_dir / "split_audit.json")
    return {
        "run_dir": str(run_dir),
        "scene": scene,
        "map_path": map_path,
        "split": split,
        "queries": int(queries) if isinstance(queries, (int, float)) else None,
        "peak_gpu_mb": float(peak_gpu_mb) if isinstance(peak_gpu_mb, (int, float)) else None,
        "latency_ms_total": dict(total_latency),
        "landmark_count": metrics.get("landmark_count"),
        "split_audit_status": split_audit.get("audit_status", "unknown") if split_audit else "unknown",
        "manifest_path": str(run_dir / "manifest.json"),
        "metrics_summary_path": str(run_dir / "metrics_summary.json"),
        "timing_profile_path": str(run_dir / "timing_profile.json"),
        "has_runtime": bool(timing.get("latency_ms")),
        "has_memory": peak_gpu_mb is not None,
    }


def _profile_score(profile: Mapping[str, Any], required_queries: int) -> tuple[int, int, int, int, str]:
    queries = profile.get("queries")
    return (
        1 if str(profile.get("split")) != "test" else 0,
        1 if profile.get("has_runtime") else 0,
        1 if profile.get("has_memory") else 0,
        1 if isinstance(queries, int) and queries >= required_queries else 0,
        str(profile.get("run_dir", "")),
    )


def _profile_index(profile_roots: Sequence[str | Path], required_queries: int) -> dict[tuple[str, str], dict[str, Any]]:
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in _discover_profile_dirs(profile_roots):
        row = _profile_row(run_dir)
        if row is None:
            continue
        key = _map_key(str(row["scene"]), str(row["map_path"]))
        existing = profiles.get(key)
        if existing is None or _profile_score(row, required_queries) > _profile_score(existing, required_queries):
            profiles[key] = row
    return profiles


def build_frozen_profile_coverage(
    frozen_manifest: Mapping[str, Any],
    *,
    profile_roots: Sequence[str | Path],
    required_queries: int = 5,
    frozen_recipe_path: str = "",
) -> dict[str, Any]:
    selected_maps = _selected_maps(frozen_manifest)
    profiles = _profile_index(profile_roots, required_queries)
    entries: list[dict[str, Any]] = []
    for selected in selected_maps:
        key = _map_key(str(selected["scene"]), str(selected["map_path"]))
        profile = profiles.get(key)
        if profile is None:
            entries.append({**selected, "profile_status": "missing", "profile": {}})
            continue
        issues: list[str] = []
        if str(profile.get("split")) == "test":
            issues.append("profile_split_is_test")
        if not profile.get("has_runtime"):
            issues.append("missing_runtime")
        if not profile.get("has_memory"):
            issues.append("missing_memory")
        queries = profile.get("queries")
        if not isinstance(queries, int) or queries < required_queries:
            issues.append("insufficient_profile_queries")
        entries.append(
            {
                **selected,
                "profile_status": "profiled" if not issues else "profiled_with_issues",
                "profile_issues": issues,
                "profile": profile,
            }
        )

    missing_count = sum(1 for entry in entries if entry["profile_status"] == "missing")
    test_split_count = sum(1 for entry in entries if entry.get("profile", {}).get("split") == "test")
    missing_runtime_count = sum(
        1 for entry in entries if entry["profile_status"] != "missing" and not entry.get("profile", {}).get("has_runtime")
    )
    missing_memory_count = sum(
        1 for entry in entries if entry["profile_status"] != "missing" and not entry.get("profile", {}).get("has_memory")
    )
    insufficient_query_count = sum(
        1
        for entry in entries
        if entry["profile_status"] != "missing"
        and (
            not isinstance(entry.get("profile", {}).get("queries"), int)
            or int(entry.get("profile", {}).get("queries")) < required_queries
        )
    )
    ready = (
        bool(entries)
        and missing_count == 0
        and test_split_count == 0
        and missing_runtime_count == 0
        and missing_memory_count == 0
        and insufficient_query_count == 0
    )
    return {
        "format": "loc_gs_frozen_profile_coverage_v1",
        "frozen_recipe_path": frozen_recipe_path,
        "freeze_id": frozen_manifest.get("freeze_id", ""),
        "profile_roots": [str(root) for root in profile_roots],
        "required_queries": required_queries,
        "unique_selected_map_count": len(entries),
        "entries": entries,
        "checks": {
            "all_unique_selected_maps_profiled": missing_count == 0 and bool(entries),
            "all_profiles_have_runtime": missing_runtime_count == 0,
            "all_profiles_have_memory": missing_memory_count == 0,
            "all_profiles_meet_required_queries": insufficient_query_count == 0,
            "no_test_split_profiles": test_split_count == 0,
            "missing_profile_count": missing_count,
            "missing_runtime_count": missing_runtime_count,
            "missing_memory_count": missing_memory_count,
            "insufficient_query_profile_count": insufficient_query_count,
            "test_split_profile_count": test_split_count,
            "profile_coverage_ready": ready,
        },
    }
