from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from loc_gs.stdloc_native.solver_aware_resampling import score_hard_query_cvar_utility


@dataclass(frozen=True)
class ReplacementDecision:
    admissible: bool
    reasons: tuple[str, ...]
    delta: dict[str, float]
    cvar: dict[str, Any] | None = None


def _source_set(source_idx: torch.Tensor | Any | None) -> set[int] | None:
    if source_idx is None:
        return None
    return {int(item) for item in torch.as_tensor(source_idx, dtype=torch.long).reshape(-1).cpu().tolist()}


def build_safe_core_from_tuple_bank(
    tuple_bank: Mapping[str, Any],
    *,
    hard_query_ids: set[int | str] | list[int | str] | tuple[int | str, ...] | None = None,
    source_idx: torch.Tensor | Any | None = None,
    top_tuples_per_query: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reserve source landmarks participating in the best hard-query viable tuples."""

    hard = {str(item) for item in hard_query_ids} if hard_query_ids is not None else None
    source = _source_set(source_idx)
    safe: set[int] = set()
    used_queries = 0
    considered_tuples = 0
    for query in tuple_bank.get("queries", []):
        query_id = str(query.get("query_id", len(safe)))
        if hard is not None and query_id not in hard:
            continue
        tuples = [
            item
            for item in query.get("tuples", [])
            if bool(item.get("viable", False))
        ]
        tuples = sorted(tuples, key=lambda item: float(item.get("geometry_logdet", 0.0)), reverse=True)
        if tuples:
            used_queries += 1
        for item in tuples[: max(0, int(top_tuples_per_query))]:
            considered_tuples += 1
            for landmark_id in item.get("landmark_ids", []):
                gid = int(landmark_id)
                if source is None or gid in source:
                    safe.add(gid)
    selected = torch.tensor(sorted(safe), dtype=torch.long)
    metadata = {
        "hard_query_count": int(len(hard)) if hard is not None else int(used_queries),
        "used_query_count": int(used_queries),
        "considered_tuple_count": int(considered_tuples),
        "safe_core_count": int(selected.numel()),
        "top_tuples_per_query": int(top_tuples_per_query),
    }
    return selected, metadata


_METRIC_ALIASES = {
    "min_eigenvalue": ("min_eigenvalue", "min_eigen", "min_eigen_H", "lambda_min"),
    "dense_worsen_risk": ("dense_worsen_risk", "dense_worsen", "dense_risk", "dense_failure_risk"),
}


def _metric(payload: Mapping[str, Any], name: str) -> float:
    for key in _METRIC_ALIASES.get(name, (name,)):
        if key in payload:
            return float(payload[key])
    return 0.0


def _normalize_query_metric_map(
    payload: Mapping[int | str, Mapping[int | str, Mapping[str, Any]]] | None,
) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = {}
    if not payload:
        return out
    for landmark_id, query_map in payload.items():
        outer = int(landmark_id)
        out[outer] = {}
        for query_id, metrics in dict(query_map).items():
            out[outer][str(query_id)] = dict(metrics)
    return out


def _query_delta(
    *,
    query_id: int | str,
    add_id: int,
    drop_id: int,
    candidate_gain: Mapping[int, Mapping[str, Mapping[str, Any]]],
    source_loss: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float]:
    query_key = str(query_id)
    gain = candidate_gain.get(int(add_id), {}).get(query_key, {})
    loss = source_loss.get(int(drop_id), {}).get(query_key, {})
    return {
        "support": _metric(gain, "support") - _metric(loss, "support"),
        "viable_tuple_mass": _metric(gain, "viable_tuple_mass") - _metric(loss, "viable_tuple_mass"),
        "logdet_H": _metric(gain, "logdet_H") - _metric(loss, "logdet_H"),
        "min_eigenvalue": _metric(gain, "min_eigenvalue") - _metric(loss, "min_eigenvalue"),
        "dense_worsen_risk": _metric(gain, "dense_worsen_risk") - _metric(loss, "dense_worsen_risk"),
        "ambiguity": _metric(gain, "ambiguity") - _metric(loss, "ambiguity"),
    }


def is_replacement_admissible(
    *,
    add_id: int,
    drop_id: int,
    hard_query_ids: set[int | str] | list[int | str] | tuple[int | str, ...],
    candidate_gain: Mapping[int | str, Mapping[int | str, Mapping[str, Any]]] | None = None,
    source_loss: Mapping[int | str, Mapping[int | str, Mapping[str, Any]]] | None = None,
    min_support_delta: float = 0.0,
    min_viable_tuple_delta: float = 0.0,
    min_logdet_delta: float = 0.0,
    min_min_eigen_delta: float = 0.0,
    max_dense_worsen_delta: float = 0.0,
    max_ambiguity_delta: float = 0.0,
    cvar_alpha: float | None = None,
    min_cvar_score: float = 0.0,
    cvar_weights: Mapping[str, float] | None = None,
    require_candidate_gain: bool = False,
) -> ReplacementDecision:
    """Check whether replacing one source landmark is solver-safe for hard queries."""

    checker = make_replacement_admissibility_checker(
        hard_query_ids=hard_query_ids,
        candidate_gain=candidate_gain,
        source_loss=source_loss,
        min_support_delta=min_support_delta,
        min_viable_tuple_delta=min_viable_tuple_delta,
        min_logdet_delta=min_logdet_delta,
        min_min_eigen_delta=min_min_eigen_delta,
        max_dense_worsen_delta=max_dense_worsen_delta,
        max_ambiguity_delta=max_ambiguity_delta,
        cvar_alpha=cvar_alpha,
        min_cvar_score=min_cvar_score,
        cvar_weights=cvar_weights,
        require_candidate_gain=require_candidate_gain,
    )
    return checker(add_id=add_id, drop_id=drop_id)


def make_replacement_admissibility_checker(
    *,
    hard_query_ids: set[int | str] | list[int | str] | tuple[int | str, ...],
    candidate_gain: Mapping[int | str, Mapping[int | str, Mapping[str, Any]]] | None = None,
    source_loss: Mapping[int | str, Mapping[int | str, Mapping[str, Any]]] | None = None,
    min_support_delta: float = 0.0,
    min_viable_tuple_delta: float = 0.0,
    min_logdet_delta: float = 0.0,
    min_min_eigen_delta: float = 0.0,
    max_dense_worsen_delta: float = 0.0,
    max_ambiguity_delta: float = 0.0,
    cvar_alpha: float | None = None,
    min_cvar_score: float = 0.0,
    cvar_weights: Mapping[str, float] | None = None,
    require_candidate_gain: bool = False,
):
    """Create a replacement checker that normalizes constraint maps once."""

    candidate_gain = _normalize_query_metric_map(candidate_gain)
    source_loss = _normalize_query_metric_map(source_loss)
    hard = [str(item) for item in hard_query_ids]

    def _check(*, add_id: int, drop_id: int) -> ReplacementDecision:
        return _replacement_admissibility_from_normalized(
            add_id=int(add_id),
            drop_id=int(drop_id),
            hard_query_ids=hard,
            candidate_gain=candidate_gain,
            source_loss=source_loss,
            min_support_delta=float(min_support_delta),
            min_viable_tuple_delta=float(min_viable_tuple_delta),
            min_logdet_delta=float(min_logdet_delta),
            min_min_eigen_delta=float(min_min_eigen_delta),
            max_dense_worsen_delta=float(max_dense_worsen_delta),
            max_ambiguity_delta=float(max_ambiguity_delta),
            cvar_alpha=cvar_alpha,
            min_cvar_score=float(min_cvar_score),
            cvar_weights=cvar_weights,
            require_candidate_gain=bool(require_candidate_gain),
        )

    return _check


def _replacement_admissibility_from_normalized(
    *,
    add_id: int,
    drop_id: int,
    hard_query_ids: list[str],
    candidate_gain: Mapping[int, Mapping[str, Mapping[str, Any]]],
    source_loss: Mapping[int, Mapping[str, Mapping[str, Any]]],
    min_support_delta: float = 0.0,
    min_viable_tuple_delta: float = 0.0,
    min_logdet_delta: float = 0.0,
    min_min_eigen_delta: float = 0.0,
    max_dense_worsen_delta: float = 0.0,
    max_ambiguity_delta: float = 0.0,
    cvar_alpha: float | None = None,
    min_cvar_score: float = 0.0,
    cvar_weights: Mapping[str, float] | None = None,
    require_candidate_gain: bool = False,
) -> ReplacementDecision:
    """Check one replacement using pre-normalized hard-query constraints."""

    hard = hard_query_ids
    aggregate = {
        "support": 0.0,
        "viable_tuple_mass": 0.0,
        "logdet_H": 0.0,
        "min_eigenvalue": 0.0,
        "dense_worsen_risk": 0.0,
        "ambiguity": 0.0,
    }
    query_deltas: dict[str, dict[str, float]] = {}
    reasons: list[str] = []
    if bool(require_candidate_gain) and not candidate_gain.get(int(add_id)):
        reasons.append("missing_candidate_gain")
    for query_id in hard:
        delta = _query_delta(
            query_id=query_id,
            add_id=int(add_id),
            drop_id=int(drop_id),
            candidate_gain=candidate_gain,
            source_loss=source_loss,
        )
        query_deltas[str(query_id)] = {key: float(value) for key, value in delta.items()}
        for key, value in delta.items():
            aggregate[key] += float(value)
        if delta["support"] < float(min_support_delta):
            reasons.append("support_drop")
        if delta["viable_tuple_mass"] < float(min_viable_tuple_delta):
            reasons.append("tuple_mass_drop")
        if delta["logdet_H"] < float(min_logdet_delta):
            reasons.append("logdet_drop")
        if delta["min_eigenvalue"] < float(min_min_eigen_delta):
            reasons.append("min_eigen_drop")
        if delta["dense_worsen_risk"] > float(max_dense_worsen_delta):
            reasons.append("dense_worsen_rise")
        if delta["ambiguity"] > float(max_ambiguity_delta):
            reasons.append("ambiguity_rise")
    cvar: dict[str, Any] | None = None
    if cvar_alpha is not None and query_deltas:
        cvar = score_hard_query_cvar_utility(
            query_deltas,
            weights=cvar_weights,
            alpha=float(cvar_alpha),
            min_score=float(min_cvar_score),
        )
        if not bool(cvar["metadata"]["accepted"]):
            reasons.append("hard_query_cvar")
    unique_reasons = tuple(sorted(set(reasons)))
    return ReplacementDecision(
        admissible=not unique_reasons,
        reasons=unique_reasons,
        delta={key: float(value) for key, value in aggregate.items()},
        cvar=cvar,
    )
