from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch

from loc_gs.stdloc_native.negative_support_memory import conflict_delta, normalize_negative_support_graph


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


CoverageSaturationMode = Literal["none", "native_percentile", "native_fraction"]


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
    *,
    saturation_targets: Mapping[str, float] | None = None,
    easy_query_gain_decay: float = 0.0,
    tail_query_ids: set[str] | None = None,
    tail_query_gain_boost: float = 1.0,
) -> tuple[float, dict[str, float]]:
    per_query: dict[str, float] = {}
    score = 0.0
    candidate_utility = tables.candidate_utility.get(int(add_id), {})
    for query_id in tables.hard_query_ids:
        utility = float(candidate_utility.get(query_id, 0.0))
        effective = float(utility)
        target = None if saturation_targets is None else saturation_targets.get(query_id)
        if target is not None and utility > 0.0:
            remaining = max(0.0, float(target) - max(0.0, float(coverage.get(query_id, 0.0))))
            capped = min(float(utility), remaining)
            excess = max(0.0, float(utility) - capped)
            effective = capped + max(0.0, float(easy_query_gain_decay)) * excess
        per_query[query_id] = float(effective)
        if target is not None and target > 0.0:
            deficit_ratio = max(0.0, float(target) - max(0.0, float(coverage.get(query_id, 0.0)))) / float(target)
            novelty = 1.0 + deficit_ratio
        else:
            novelty = 1.0 / math.sqrt(1.0 + max(0.0, float(coverage.get(query_id, 0.0))))
        if tail_query_ids is not None and query_id in tail_query_ids:
            novelty *= 1.0 + max(0.0, float(tail_query_gain_boost))
        score += float(effective) * novelty
    return float(score - _candidate_threshold_penalty(add_id, tables)), per_query


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    qf = min(1.0, max(0.0, float(q)))
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = qf * float(len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(ordered[lo])
    weight = position - float(lo)
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def _build_saturation_targets(
    coverage: Mapping[str, float],
    *,
    mode: CoverageSaturationMode | str = "none",
    percentile: float = 0.75,
    fraction: float = 1.0,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    mode_text = str(mode or "none")
    if mode_text == "none":
        return None, {
            "enabled": False,
            "mode": "none",
            "target_count": 0,
            "saturated_query_count": 0,
        }
    if mode_text not in {"native_percentile", "native_fraction"}:
        raise ValueError("coverage_saturation_mode must be one of: none, native_percentile, native_fraction")
    if mode_text == "native_percentile":
        target_value = _percentile(list(coverage.values()), float(percentile))
        targets = {str(query_id): float(target_value) for query_id in coverage}
    else:
        frac = max(0.0, float(fraction))
        targets = {str(query_id): max(0.0, float(value) * frac) for query_id, value in coverage.items()}
    saturated_count = sum(1 for query_id, target in targets.items() if float(coverage.get(query_id, 0.0)) >= float(target))
    finite_targets = list(targets.values())
    return targets, {
        "enabled": True,
        "mode": mode_text,
        "target_count": int(len(targets)),
        "saturated_query_count": int(saturated_count),
        "target_min": float(min(finite_targets)) if finite_targets else 0.0,
        "target_median": _percentile(finite_targets, 0.5),
        "target_max": float(max(finite_targets)) if finite_targets else 0.0,
        "percentile": float(percentile),
        "fraction": float(fraction),
    }


def _tail_query_ids(
    coverage: Mapping[str, float],
    targets: Mapping[str, float] | None,
    *,
    alpha: float,
) -> set[str]:
    alpha_f = max(0.0, min(1.0, float(alpha)))
    if alpha_f <= 0.0:
        return set()
    ratios: list[tuple[float, str]] = []
    for query_id, value in coverage.items():
        target = None if targets is None else targets.get(query_id)
        denom = float(target) if target is not None and float(target) > 0.0 else max(1.0, float(value))
        ratios.append((float(value) / denom, str(query_id)))
    if not ratios:
        return set()
    count = max(1, int(math.ceil(alpha_f * len(ratios))))
    return {query_id for _, query_id in sorted(ratios)[:count]}


def _drop_cost(
    *,
    drop_id: int,
    add_per_query: Mapping[str, float],
    coverage: Mapping[str, float],
    tables: CoverageTables,
    utility_t: torch.Tensor,
    min_query_coverage: float,
    saturation_targets: Mapping[str, float] | None = None,
    tail_query_ids: set[str] | None = None,
    saturation_drop_protection_weight: float = 0.0,
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
        if saturation_targets is not None and float(saturation_drop_protection_weight) > 0.0:
            target = float(saturation_targets.get(query_id, 0.0))
            if target > 0.0 and projected < target:
                deficit_ratio = (target - projected) / target
                tail_boost = 2.0 if tail_query_ids is not None and query_id in tail_query_ids else 1.0
                cost += float(saturation_drop_protection_weight) * tail_boost * drop_contrib * deficit_ratio
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
    coverage_saturation_mode: CoverageSaturationMode | str = "none",
    coverage_saturation_percentile: float = 0.75,
    coverage_saturation_fraction: float = 1.0,
    easy_query_gain_decay: float = 0.0,
    tail_cvar_alpha: float = 0.0,
    tail_query_gain_boost: float = 1.0,
    hard_query_min_gain: float = 0.0,
    saturation_drop_protection_weight: float = 0.0,
    negative_support_graph: Mapping[str, Any] | None = None,
    negative_conflict_pair_weight: float = 0.0,
    negative_conflict_unary_weight: float = 0.0,
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
    saturation_targets, saturation_metadata = _build_saturation_targets(
        coverage,
        mode=coverage_saturation_mode,
        percentile=float(coverage_saturation_percentile),
        fraction=float(coverage_saturation_fraction),
    )
    tail_queries = _tail_query_ids(coverage, saturation_targets, alpha=float(tail_cvar_alpha))
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
    negative_graph = normalize_negative_support_graph(negative_support_graph)
    conflict_enabled = (
        bool(negative_graph.get("edge_weights") or negative_graph.get("unary_risk"))
        and (float(negative_conflict_pair_weight) > 0.0 or float(negative_conflict_unary_weight) > 0.0)
    )
    conflict_penalty_sum = 0.0

    while candidate_set and removable and len(edits) < max_edits_i:
        scored_candidates: list[tuple[float, float, int, dict[str, float]]] = []
        for add_id in candidate_set:
            coverage_gain, per_query = _candidate_coverage_gain(
                add_id,
                coverage,
                coverage_tables,
                saturation_targets=saturation_targets,
                easy_query_gain_decay=float(easy_query_gain_decay),
                tail_query_ids=tail_queries,
                tail_query_gain_boost=float(tail_query_gain_boost),
            )
            if coverage_gain < float(min_coverage_gain):
                continue
            undercovered_gain = 0.0
            if saturation_targets is not None:
                for query_id, gain in per_query.items():
                    if float(coverage.get(query_id, 0.0)) < float(saturation_targets.get(query_id, 0.0)):
                        undercovered_gain += max(0.0, float(gain))
            else:
                undercovered_gain = sum(max(0.0, float(value)) for value in per_query.values())
            if undercovered_gain < float(hard_query_min_gain):
                continue
            add_conflict = (
                max(
                    0.0,
                    conflict_delta(
                        add_id=int(add_id),
                        drop_id=None,
                        selected_ids=selected,
                        graph=negative_graph,
                        pair_weight=float(negative_conflict_pair_weight),
                        unary_weight=float(negative_conflict_unary_weight),
                    ),
                )
                if conflict_enabled
                else 0.0
            )
            total_score = float(coverage_gain) + float(utility_t[int(add_id)].item()) - float(add_conflict)
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
                    saturation_targets=saturation_targets,
                    tail_query_ids=tail_queries,
                    saturation_drop_protection_weight=float(saturation_drop_protection_weight),
                )
                if protected:
                    protected_drop_count += 1
                drop_options.append((cost, int(drop_id)))
            drop_cost_evaluations += int(len(drop_options))
            for cost, drop_id in sorted(drop_options)[:drop_scan_limit]:
                drop_scan_attempts += 1
                coverage_gain = -float(neg_coverage_gain)
                utility_gain = float(utility_t[int(add_id)].item()) - float(utility_t[int(drop_id)].item())
                conflict_cost = (
                    max(
                        0.0,
                        conflict_delta(
                            add_id=int(add_id),
                            drop_id=int(drop_id),
                            selected_ids=selected,
                            graph=negative_graph,
                            pair_weight=float(negative_conflict_pair_weight),
                            unary_weight=float(negative_conflict_unary_weight),
                        ),
                    )
                    if conflict_enabled
                    else 0.0
                )
                replacement_gain = float(coverage_gain) + float(utility_gain) - float(cost) - float(conflict_cost)
                if replacement_gain <= 0.0:
                    rejected_low_gain += 1
                    continue
                if is_admissible is not None and not bool(is_admissible(add_id, drop_id)):
                    rejected_by_admissibility += 1
                    continue
                accepted = (
                    int(add_id),
                    int(drop_id),
                    float(coverage_gain),
                    float(replacement_gain),
                    add_per_query,
                    float(conflict_cost),
                )
                break
            if accepted is not None:
                break
            candidate_set.remove(int(add_id))
        if accepted is None:
            break
        add_id, drop_id, coverage_gain, replacement_gain, _, conflict_cost = accepted
        conflict_penalty_sum += float(conflict_cost)
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
                "negative_conflict_penalty": float(conflict_cost),
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
        "coverage_saturation": saturation_metadata,
        "easy_query_gain_decay": float(easy_query_gain_decay),
        "tail_cvar_alpha": float(tail_cvar_alpha),
        "tail_query_gain_boost": float(tail_query_gain_boost),
        "tail_query_count": int(len(tail_queries)),
        "hard_query_min_gain": float(hard_query_min_gain),
        "hard_query_count": int(len(coverage_tables.hard_query_ids)),
        "negative_support_graph": {
            "enabled": bool(conflict_enabled),
            "edge_count": int(negative_graph.get("edge_count", 0) or 0),
            "unary_count": int(len(negative_graph.get("unary_risk", {}))),
            "pair_weight": float(negative_conflict_pair_weight),
            "unary_weight": float(negative_conflict_unary_weight),
            "applied_penalty_sum": float(conflict_penalty_sum),
        },
        "initial_query_coverage": initial_coverage,
        "final_query_coverage": {key: float(value) for key, value in coverage.items()},
    }
    metadata["coverage_saturation"]["drop_protection_weight"] = float(saturation_drop_protection_weight)
    return {
        "sampled_idx": sampled,
        "edits": edits,
        "metadata": metadata,
    }
