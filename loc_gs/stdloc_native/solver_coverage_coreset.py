from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


_METRIC_FIELDS = (
    "support",
    "viable_tuple_mass",
    "logdet_H",
    "min_eigenvalue",
    "dense_worsen_risk",
    "ambiguity",
)
_METRIC_ALIASES = {
    "min_eigenvalue": ("min_eigenvalue", "min_eigen", "min_eigen_H", "lambda_min"),
    "dense_worsen_risk": ("dense_worsen_risk", "dense_worsen", "dense_risk", "dense_failure_risk"),
}
_DEFAULT_WEIGHTS = {
    "support": 1.0,
    "viable_tuple_mass": 1.0,
    "logdet_H": 1.0,
    "min_eigenvalue": 1.0,
    "dense_worsen_risk": -1.0,
    "ambiguity": -1.0,
}


@dataclass(frozen=True)
class CoverageTables:
    hard_query_ids: tuple[str, ...]
    candidate_gain: dict[int, dict[str, dict[str, float]]]
    source_loss: dict[int, dict[str, dict[str, float]]]
    candidate_utility: dict[int, dict[str, float]]
    source_utility: dict[int, dict[str, float]]
    weights: dict[str, float]
    thresholds: dict[str, Any]


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


def _canonical_metric(name: str) -> str:
    for canonical, aliases in _METRIC_ALIASES.items():
        if name in aliases:
            return canonical
    return str(name)


def _metric(payload: Mapping[str, Any], name: str) -> float:
    for key in _METRIC_ALIASES.get(name, (name,)):
        if key in payload:
            return float(payload[key])
    return 0.0


def _metric_utility(metrics: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return float(sum(float(weights[field]) * _metric(metrics, field) for field in _METRIC_FIELDS))


def _normalize_metric_map(payload: Any) -> dict[int, dict[str, dict[str, float]]]:
    out: dict[int, dict[str, dict[str, float]]] = {}
    if not payload:
        return out
    if not isinstance(payload, Mapping):
        raise TypeError("solver coverage metric maps must be mappings")
    for raw_landmark, raw_query_map in payload.items():
        landmark_id = int(raw_landmark)
        out[landmark_id] = {}
        for raw_query, raw_metrics in dict(raw_query_map).items():
            metrics: dict[str, float] = {}
            for raw_name, raw_value in dict(raw_metrics).items():
                canonical = _canonical_metric(str(raw_name))
                if canonical in _METRIC_FIELDS:
                    metrics[canonical] = float(raw_value)
            out[landmark_id][str(raw_query)] = metrics
    return out


def _query_utility_map(
    payload: Mapping[int, Mapping[str, Mapping[str, float]]],
    weights: Mapping[str, float],
) -> dict[int, dict[str, float]]:
    return {
        int(landmark_id): {
            str(query_id): _metric_utility(metrics, weights)
            for query_id, metrics in query_map.items()
        }
        for landmark_id, query_map in payload.items()
    }


def build_solver_coverage_tables(payload: Mapping[str, Any]) -> CoverageTables:
    """Normalize solver-admissibility JSON into coverage-aware edit tables."""

    if not isinstance(payload, Mapping):
        raise TypeError("solver coverage payload must be a mapping")
    thresholds = dict(payload.get("thresholds", {}))
    raw_weights = thresholds.get("cvar_weights", thresholds.get("hard_query_cvar_weights", {}))
    weights = dict(_DEFAULT_WEIGHTS)
    if isinstance(raw_weights, Mapping):
        for raw_name, raw_value in raw_weights.items():
            canonical = _canonical_metric(str(raw_name))
            if canonical in weights:
                weights[canonical] = float(raw_value)
    candidate_gain = _normalize_metric_map(payload.get("candidate_gain", {}))
    source_loss = _normalize_metric_map(payload.get("source_loss", {}))
    return CoverageTables(
        hard_query_ids=tuple(str(item) for item in payload.get("hard_query_ids", [])),
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        candidate_utility=_query_utility_map(candidate_gain, weights),
        source_utility=_query_utility_map(source_loss, weights),
        weights=weights,
        thresholds=thresholds,
    )


def _initial_query_coverage(
    selected: set[int],
    tables: CoverageTables,
) -> dict[str, float]:
    coverage = {query_id: 0.0 for query_id in tables.hard_query_ids}
    for landmark_id in selected:
        for query_id, value in tables.source_utility.get(int(landmark_id), {}).items():
            if query_id in coverage:
                coverage[query_id] += max(0.0, float(value))
    return coverage


def _threshold(thresholds: Mapping[str, Any], name: str, default: Any, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in thresholds:
            return thresholds[key]
    return default


def _candidate_threshold_penalty(add_id: int, tables: CoverageTables) -> float:
    query_map = tables.candidate_gain.get(int(add_id), {})
    dense_risk = sum(max(0.0, _metric(metrics, "dense_worsen_risk")) for metrics in query_map.values())
    ambiguity = sum(max(0.0, _metric(metrics, "ambiguity")) for metrics in query_map.values())
    max_dense = float(_threshold(tables.thresholds, "max_dense_worsen_delta", 0.0, "max_dense_worsen_risk_delta"))
    max_ambiguity = float(_threshold(tables.thresholds, "max_ambiguity_delta", 0.0))
    penalty = 0.0
    if dense_risk > max_dense:
        penalty += 8.0 * (dense_risk - max_dense)
    if ambiguity > max_ambiguity:
        penalty += 4.0 * (ambiguity - max_ambiguity)
    return float(penalty)


def _candidate_coverage_gain(
    add_id: int,
    coverage: Mapping[str, float],
    tables: CoverageTables,
) -> tuple[float, dict[str, float]]:
    per_query: dict[str, float] = {}
    score = 0.0
    candidate_utility = tables.candidate_utility.get(int(add_id), {})
    for query_id in tables.hard_query_ids:
        utility = float(candidate_utility.get(query_id, 0.0))
        per_query[query_id] = float(utility)
        novelty = 1.0 / math.sqrt(1.0 + max(0.0, float(coverage.get(query_id, 0.0))))
        score += float(utility) * novelty
    return float(score - _candidate_threshold_penalty(add_id, tables)), per_query


def _drop_cost(
    *,
    drop_id: int,
    add_per_query: Mapping[str, float],
    coverage: Mapping[str, float],
    tables: CoverageTables,
    utility_t: torch.Tensor,
    min_query_coverage: float,
) -> tuple[float, bool]:
    cost = float(utility_t[int(drop_id)].item())
    protected = False
    overlap_reward = 0.0
    for query_id in tables.hard_query_ids:
        drop_contrib = max(
            0.0,
            float(tables.source_utility.get(int(drop_id), {}).get(query_id, 0.0)),
        )
        if drop_contrib <= 0.0:
            continue
        add_contrib = max(0.0, float(add_per_query.get(query_id, 0.0)))
        overlap_reward += min(add_contrib, drop_contrib)
        projected = float(coverage.get(query_id, 0.0)) + add_contrib - drop_contrib
        if projected < float(min_query_coverage):
            protected = True
            cost += 10.0 * (float(min_query_coverage) - projected + drop_contrib)
    cost -= 0.001 * overlap_reward
    return float(cost), protected


def _apply_coverage_edit(
    coverage: dict[str, float],
    *,
    add_id: int,
    drop_id: int,
    tables: CoverageTables,
) -> None:
    for query_id in tables.hard_query_ids:
        add_value = max(
            0.0,
            float(tables.candidate_utility.get(int(add_id), {}).get(query_id, 0.0)),
        )
        drop_value = max(
            0.0,
            float(tables.source_utility.get(int(drop_id), {}).get(query_id, 0.0)),
        )
        coverage[query_id] = max(0.0, float(coverage.get(query_id, 0.0)) + add_value - drop_value)


def _source_contrib_by_query(tables: CoverageTables) -> dict[str, list[tuple[float, int]]]:
    by_query: dict[str, list[tuple[float, int]]] = {query_id: [] for query_id in tables.hard_query_ids}
    for source_id, query_map in tables.source_utility.items():
        for query_id in tables.hard_query_ids:
            contrib = max(0.0, float(query_map.get(query_id, 0.0)))
            if contrib > 0.0:
                by_query[query_id].append((-float(contrib), int(source_id)))
    for query_id in by_query:
        by_query[query_id].sort()
    return by_query


def _candidate_drop_ids(
    *,
    removable: list[tuple[float, int]],
    removable_ids: set[int],
    add_per_query: Mapping[str, float],
    source_by_query: Mapping[str, list[tuple[float, int]]],
    max_drop_scan: int,
) -> list[int]:
    base_window = max(32, int(max_drop_scan) * 8)
    same_query_window = max(16, int(max_drop_scan) * 4)
    drop_ids: list[int] = []
    used: set[int] = set()
    for _, drop_id in removable[:base_window]:
        gid = int(drop_id)
        drop_ids.append(gid)
        used.add(gid)
    for query_id, add_value in add_per_query.items():
        if float(add_value) <= 0.0:
            continue
        added_for_query = 0
        for _, source_id in source_by_query.get(str(query_id), []):
            gid = int(source_id)
            if gid not in removable_ids or gid in used:
                continue
            drop_ids.append(gid)
            used.add(gid)
            added_for_query += 1
            if added_for_query >= same_query_window:
                break
    return drop_ids


def solver_coverage_local_edit(
    *,
    source_idx: torch.Tensor | Any,
    candidate_pool: torch.Tensor | Any,
    utility: torch.Tensor | Any,
    evidence_mask: torch.Tensor | Any,
    coverage_tables: CoverageTables,
    safe_core: torch.Tensor | Any | None = None,
    is_admissible: Callable[[int, int], bool] | None = None,
    max_edits: int = 256,
    max_drop_scan: int = 1,
    min_coverage_gain: float = 0.0,
    min_query_coverage: float = 1.0,
) -> dict[str, Any]:
    """Same-budget local replacement with hard-query solver coverage scoring."""

    source = _long_vector(source_idx, name="source_idx")
    pool = _long_vector(candidate_pool, name="candidate_pool")
    utility_t = _float_vector(utility, name="utility")
    evidence = torch.as_tensor(evidence_mask, dtype=torch.bool).reshape(-1).cpu()
    if evidence.shape[0] != utility_t.shape[0]:
        raise ValueError("evidence_mask and utility must have the same length")
    safe = set(_long_vector(safe_core, name="safe_core").tolist()) if safe_core is not None else set()
    selected: set[int] = {int(item) for item in source.tolist() if 0 <= int(item) < utility_t.numel()}
    source_set = set(selected)
    candidate_set = {
        int(item)
        for item in pool.tolist()
        if 0 <= int(item) < utility_t.numel() and int(item) not in selected and bool(evidence[int(item)].item())
    }
    removable = sorted(
        (float(utility_t[int(item)].item()), int(item))
        for item in selected
        if int(item) not in safe
    )
    coverage = _initial_query_coverage(selected, coverage_tables)
    source_by_query = _source_contrib_by_query(coverage_tables)
    initial_coverage = dict(coverage)
    edits: list[dict[str, Any]] = []
    rejected_by_admissibility = 0
    rejected_low_gain = 0
    protected_drop_count = 0
    drop_scan_attempts = 0
    drop_cost_evaluations = 0
    drop_scan_limit = max(1, int(max_drop_scan))
    max_edits_i = max(0, int(max_edits))

    while candidate_set and removable and len(edits) < max_edits_i:
        scored_candidates: list[tuple[float, float, int, dict[str, float]]] = []
        for add_id in candidate_set:
            coverage_gain, per_query = _candidate_coverage_gain(add_id, coverage, coverage_tables)
            if coverage_gain < float(min_coverage_gain):
                continue
            total_score = float(coverage_gain) + float(utility_t[int(add_id)].item())
            scored_candidates.append((-total_score, -coverage_gain, int(add_id), per_query))
        if not scored_candidates:
            rejected_low_gain += int(len(candidate_set))
            break
        accepted: tuple[int, int, float, float, dict[str, float]] | None = None
        for _, neg_coverage_gain, add_id, add_per_query in sorted(scored_candidates):
            drop_options: list[tuple[float, int]] = []
            removable_ids = {int(drop_id) for _, drop_id in removable}
            for drop_id in _candidate_drop_ids(
                removable=removable,
                removable_ids=removable_ids,
                add_per_query=add_per_query,
                source_by_query=source_by_query,
                max_drop_scan=drop_scan_limit,
            ):
                cost, protected = _drop_cost(
                    drop_id=int(drop_id),
                    add_per_query=add_per_query,
                    coverage=coverage,
                    tables=coverage_tables,
                    utility_t=utility_t,
                    min_query_coverage=float(min_query_coverage),
                )
                if protected:
                    protected_drop_count += 1
                drop_options.append((cost, int(drop_id)))
            drop_cost_evaluations += int(len(drop_options))
            for cost, drop_id in sorted(drop_options)[:drop_scan_limit]:
                drop_scan_attempts += 1
                coverage_gain = -float(neg_coverage_gain)
                utility_gain = float(utility_t[int(add_id)].item()) - float(utility_t[int(drop_id)].item())
                replacement_gain = float(coverage_gain) + float(utility_gain) - float(cost)
                if replacement_gain <= 0.0:
                    rejected_low_gain += 1
                    continue
                if is_admissible is not None and not bool(is_admissible(add_id, drop_id)):
                    rejected_by_admissibility += 1
                    continue
                accepted = (int(add_id), int(drop_id), float(coverage_gain), float(replacement_gain), add_per_query)
                break
            if accepted is not None:
                break
            candidate_set.remove(int(add_id))
        if accepted is None:
            break
        add_id, drop_id, coverage_gain, replacement_gain, _ = accepted
        selected.remove(drop_id)
        selected.add(add_id)
        candidate_set.discard(add_id)
        for index, (_, removable_id) in enumerate(removable):
            if int(removable_id) == int(drop_id):
                removable.pop(index)
                break
        if add_id not in safe:
            bisect.insort(removable, (float(utility_t[int(add_id)].item()), int(add_id)))
        _apply_coverage_edit(coverage, add_id=add_id, drop_id=drop_id, tables=coverage_tables)
        edits.append(
            {
                "add_id": int(add_id),
                "drop_id": int(drop_id),
                "coverage_gain": float(coverage_gain),
                "replacement_gain": float(replacement_gain),
                "utility_gain": float(utility_t[int(add_id)].item()) - float(utility_t[int(drop_id)].item()),
            }
        )

    source_rank = {int(gid): rank for rank, gid in enumerate(source.tolist())}
    ordered = sorted(
        selected,
        key=lambda gid: (source_rank.get(int(gid), len(source_rank)), -float(utility_t[int(gid)].item()), int(gid)),
    )
    sampled = torch.tensor(ordered, dtype=torch.long)
    safe_core_dropped = len(safe & source_set - set(sampled.tolist()))
    metadata = {
        "coverage_policy": "solver_coverage_coreset",
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
        "coverage_protected_drop_count": int(protected_drop_count),
        "drop_scan_attempts": int(drop_scan_attempts),
        "drop_cost_evaluations": int(drop_cost_evaluations),
        "max_drop_scan": int(drop_scan_limit),
        "max_edits": int(max_edits_i),
        "min_coverage_gain": float(min_coverage_gain),
        "min_query_coverage": float(min_query_coverage),
        "hard_query_count": int(len(coverage_tables.hard_query_ids)),
        "initial_query_coverage": initial_coverage,
        "final_query_coverage": {key: float(value) for key, value in coverage.items()},
    }
    return {
        "sampled_idx": sampled,
        "edits": edits,
        "metadata": metadata,
    }
