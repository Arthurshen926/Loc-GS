from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping


def _landmark_id(record: Mapping[str, Any]) -> int | None:
    for key in ("landmark_id", "matched_landmark_id", "gaussian_id", "matched_gaussian_id"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        try:
            gid = int(raw)
        except (TypeError, ValueError):
            continue
        return gid if gid >= 0 else None
    return None


def build_negative_support_graph(
    records: Iterable[Mapping[str, Any]],
    *,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Build pairwise landmark conflicts from hard-negative feedback records."""

    grouped: dict[tuple[str, str], list[tuple[int, bool, float]]] = defaultdict(list)
    unary: dict[int, float] = defaultdict(float)
    for record in records:
        gid = _landmark_id(record)
        if gid is None:
            continue
        score = float(record.get("score", record.get("descriptor_score", 1.0)) or 0.0)
        if score < float(min_score):
            continue
        inlier = bool(record.get("pnp_inlier", record.get("pair_label", False)))
        query_id = str(record.get("query_id", record.get("image_id", "")))
        keypoint_id = str(record.get("keypoint_id", record.get("match_keypoint_id", "")))
        grouped[(query_id, keypoint_id)].append((int(gid), inlier, max(score, 0.0)))
        if not inlier:
            unary[int(gid)] += max(score, 0.0)

    edges: dict[tuple[int, int], float] = defaultdict(float)
    for rows in grouped.values():
        negatives = [(gid, score) for gid, inlier, score in rows if not inlier]
        if len(negatives) < 2:
            continue
        for (a, score_a), (b, score_b) in combinations(sorted(negatives), 2):
            key = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
            edges[key] += min(float(score_a), float(score_b))
    return {
        "format": "loc_gs_negative_support_memory_v1",
        "edge_weights": dict(edges),
        "unary_risk": dict(unary),
        "group_count": int(len(grouped)),
        "edge_count": int(len(edges)),
    }


def _edge_tuple(edge: Any) -> tuple[int, int]:
    if isinstance(edge, str):
        left, right = edge.split(":", 1)
        a, b = int(left), int(right)
    else:
        a, b = edge
        a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)


def normalize_negative_support_graph(graph: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize graph edge keys for runtime selection code."""

    if not graph:
        return {"format": "loc_gs_negative_support_memory_v1", "edge_weights": {}, "unary_risk": {}, "edge_count": 0}
    if isinstance(graph, dict) and graph.get("_normalized") is True:
        return graph
    edge_weights = {
        _edge_tuple(edge): float(weight)
        for edge, weight in dict(graph.get("edge_weights", {})).items()
    }
    unary_risk = {int(gid): float(weight) for gid, weight in dict(graph.get("unary_risk", {})).items()}
    adjacency: dict[int, dict[int, float]] = defaultdict(dict)
    for (a, b), weight in edge_weights.items():
        adjacency[int(a)][int(b)] = adjacency[int(a)].get(int(b), 0.0) + float(weight)
        adjacency[int(b)][int(a)] = adjacency[int(b)].get(int(a), 0.0) + float(weight)
    out = dict(graph)
    out["edge_weights"] = edge_weights
    out["unary_risk"] = unary_risk
    out["adjacency"] = {int(gid): dict(neighbors) for gid, neighbors in adjacency.items()}
    out["edge_count"] = int(len(edge_weights))
    out["_normalized"] = True
    return out


def conflict_penalty(selected_ids: Iterable[int], graph: Mapping[str, Any], *, pair_weight: float = 1.0, unary_weight: float = 0.0) -> float:
    """Compute selected-set conflict penalty from a negative support graph."""

    selected = {int(item) for item in selected_ids}
    normalized = normalize_negative_support_graph(graph)
    edge_weights = normalized.get("edge_weights", {})
    unary_risk = normalized.get("unary_risk", {})
    penalty = 0.0
    for edge, weight in edge_weights.items():
        a, b = _edge_tuple(edge)
        if int(a) in selected and int(b) in selected:
            penalty += float(pair_weight) * float(weight)
    if float(unary_weight) > 0.0:
        for gid in selected:
            penalty += float(unary_weight) * float(unary_risk.get(gid, 0.0))
    return float(penalty)


def conflict_delta(
    *,
    add_id: int,
    drop_id: int | None,
    selected_ids: Iterable[int],
    graph: Mapping[str, Any],
    pair_weight: float = 1.0,
    unary_weight: float = 0.0,
) -> float:
    """Incremental conflict penalty for replacing drop_id with add_id."""

    selected = selected_ids if isinstance(selected_ids, set) else {int(item) for item in selected_ids}
    normalized = normalize_negative_support_graph(graph)
    adjacency = normalized.get("adjacency", {})
    unary_risk = normalized.get("unary_risk", {})
    add = int(add_id)
    drop = int(drop_id) if drop_id is not None else None
    if drop is not None and add == drop:
        return 0.0

    unary_delta = 0.0
    if float(unary_weight) > 0.0:
        if add not in selected:
            unary_delta += float(unary_weight) * float(unary_risk.get(add, 0.0))
        if drop is not None and drop in selected:
            unary_delta -= float(unary_weight) * float(unary_risk.get(drop, 0.0))

    pair_delta = 0.0
    if float(pair_weight) > 0.0:
        for other, weight in dict(adjacency.get(add, {})).items():
            other_id = int(other)
            if other_id in selected and other_id != drop:
                pair_delta += float(pair_weight) * float(weight)
        if drop is not None and drop in selected:
            for other, weight in dict(adjacency.get(drop, {})).items():
                other_id = int(other)
                if other_id in selected and other_id != drop:
                    pair_delta -= float(pair_weight) * float(weight)
    return float(unary_delta + pair_delta)
