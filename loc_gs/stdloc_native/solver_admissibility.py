from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ReplacementDecision:
    admissible: bool
    reasons: tuple[str, ...]
    delta: dict[str, float]


def _source_set(source_idx: torch.Tensor | Any | None) -> set[int] | None:
    if source_idx is None:
        return None
    return {int(item) for item in torch.as_tensor(source_idx, dtype=torch.long).reshape(-1).cpu().tolist()}


def build_safe_core_from_tuple_bank(
    tuple_bank: Mapping[str, Any],
    *,
    hard_query_ids: set[int] | list[int] | tuple[int, ...] | None = None,
    source_idx: torch.Tensor | Any | None = None,
    top_tuples_per_query: int = 4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reserve source landmarks participating in the best hard-query viable tuples."""

    hard = {int(item) for item in hard_query_ids} if hard_query_ids is not None else None
    source = _source_set(source_idx)
    safe: set[int] = set()
    used_queries = 0
    considered_tuples = 0
    for query in tuple_bank.get("queries", []):
        query_id = int(query.get("query_id", len(safe)))
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


def _metric(payload: Mapping[str, Any], name: str) -> float:
    return float(payload.get(name, 0.0))


def _query_delta(
    *,
    query_id: int,
    add_id: int,
    drop_id: int,
    candidate_gain: Mapping[int, Mapping[int, Mapping[str, Any]]],
    source_loss: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[str, float]:
    gain = candidate_gain.get(int(add_id), {}).get(int(query_id), {})
    loss = source_loss.get(int(drop_id), {}).get(int(query_id), {})
    return {
        "support": _metric(gain, "support") - _metric(loss, "support"),
        "viable_tuple_mass": _metric(gain, "viable_tuple_mass") - _metric(loss, "viable_tuple_mass"),
        "logdet_H": _metric(gain, "logdet_H") - _metric(loss, "logdet_H"),
        "ambiguity": _metric(gain, "ambiguity") - _metric(loss, "ambiguity"),
    }


def is_replacement_admissible(
    *,
    add_id: int,
    drop_id: int,
    hard_query_ids: set[int] | list[int] | tuple[int, ...],
    candidate_gain: Mapping[int, Mapping[int, Mapping[str, Any]]] | None = None,
    source_loss: Mapping[int, Mapping[int, Mapping[str, Any]]] | None = None,
    min_support_delta: float = 0.0,
    min_viable_tuple_delta: float = 0.0,
    min_logdet_delta: float = 0.0,
    max_ambiguity_delta: float = 0.0,
    require_candidate_gain: bool = False,
) -> ReplacementDecision:
    """Check whether replacing one source landmark is solver-safe for hard queries."""

    candidate_gain = candidate_gain or {}
    source_loss = source_loss or {}
    hard = [int(item) for item in hard_query_ids]
    aggregate = {"support": 0.0, "viable_tuple_mass": 0.0, "logdet_H": 0.0, "ambiguity": 0.0}
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
        for key, value in delta.items():
            aggregate[key] += float(value)
        if delta["support"] < float(min_support_delta):
            reasons.append("support_drop")
        if delta["viable_tuple_mass"] < float(min_viable_tuple_delta):
            reasons.append("tuple_mass_drop")
        if delta["logdet_H"] < float(min_logdet_delta):
            reasons.append("logdet_drop")
        if delta["ambiguity"] > float(max_ambiguity_delta):
            reasons.append("ambiguity_rise")
    unique_reasons = tuple(sorted(set(reasons)))
    return ReplacementDecision(
        admissible=not unique_reasons,
        reasons=unique_reasons,
        delta={key: float(value) for key, value in aggregate.items()},
    )
