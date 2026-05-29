from __future__ import annotations

from typing import Any, Mapping, Sequence

from loc_gs.stdloc_native.negative_support_memory import conflict_delta


def _query_value(table: Mapping[int, Mapping[str, float]] | Mapping[int, float] | None, gid: int, query_id: str) -> float:
    if not table:
        return 0.0
    raw = table.get(int(gid), 0.0)  # type: ignore[arg-type]
    if isinstance(raw, Mapping):
        return float(raw.get(str(query_id), 0.0) or 0.0)
    return float(raw or 0.0)


def _sum_query_values(table: Mapping[int, Mapping[str, float]] | Mapping[int, float] | None, gid: int, query_ids: set[str]) -> float:
    return float(sum(_query_value(table, int(gid), query_id) for query_id in query_ids))


def _source_loss_value(source_loss: Mapping[int, float] | Mapping[str, float] | None, gid: int) -> float:
    if not source_loss:
        return 0.0
    return float(source_loss.get(int(gid), source_loss.get(str(gid), 0.0)) or 0.0)  # type: ignore[arg-type]


def select_failure_aware_replacements(
    *,
    source_ids: Sequence[int],
    candidate_ids: Sequence[int],
    query_baseline_dense_te_cm: Mapping[str, float],
    candidate_query_gain: Mapping[int, Mapping[str, float]],
    candidate_regression_risk: Mapping[int, Mapping[str, float]] | None,
    dense_worsened_query_ids: Sequence[str],
    source_loss: Mapping[int, float] | Mapping[str, float] | None = None,
    dense_worsen_risk: Mapping[int, float] | None = None,
    ambiguity_risk: Mapping[int, float] | None = None,
    negative_support_graph: Mapping[str, Any] | None = None,
    protected_source_ids: Sequence[int] | None = None,
    max_edits: int = 128,
    protected_te_cm: float = 15.0,
    hard_te_cm: float = 20.0,
    max_protected_regression_cm: float = 0.0,
    dense_worsen_weight: float = 1.0,
    ambiguity_weight: float = 1.0,
    conflict_weight: float = 1.0,
) -> dict[str, Any]:
    """Select support edits with explicit hard-query gain and native do-no-harm constraints."""

    selected = {int(item) for item in source_ids}
    protected_sources = {int(item) for item in protected_source_ids or []}
    protected_queries = {
        str(query_id)
        for query_id, te in query_baseline_dense_te_cm.items()
        if float(te) < float(protected_te_cm)
    }
    hard_queries = {
        str(query_id)
        for query_id, te in query_baseline_dense_te_cm.items()
        if float(te) > float(hard_te_cm)
    } | {str(query_id) for query_id in dense_worsened_query_ids}
    if not hard_queries:
        hard_queries = {str(query_id) for query_id in query_baseline_dense_te_cm}

    rejected_protected = 0
    scored: list[tuple[float, int, float]] = []
    for add_id in candidate_ids:
        add = int(add_id)
        if add in selected:
            continue
        protected_risk = _sum_query_values(candidate_regression_risk, add, protected_queries)
        if protected_risk > float(max_protected_regression_cm):
            rejected_protected += 1
            continue
        hard_gain = _sum_query_values(candidate_query_gain, add, hard_queries)
        if hard_gain <= 0.0:
            continue
        dense_penalty = float(dense_worsen_weight) * float((dense_worsen_risk or {}).get(add, 0.0))
        ambiguity_penalty = float(ambiguity_weight) * float((ambiguity_risk or {}).get(add, 0.0))
        conflict_penalty = 0.0
        if negative_support_graph:
            conflict_penalty = conflict_delta(
                add_id=add,
                drop_id=None,
                selected_ids=selected,
                graph=negative_support_graph,
                pair_weight=float(conflict_weight),
                unary_weight=0.0,
            )
        score = float(hard_gain - protected_risk - dense_penalty - ambiguity_penalty - conflict_penalty)
        if score > 0.0:
            scored.append((score, add, hard_gain))

    scored.sort(key=lambda item: (-item[0], item[1]))
    drops = sorted(
        (int(gid) for gid in selected if int(gid) not in protected_sources),
        key=lambda gid: (-_source_loss_value(source_loss, gid), -gid),
    )
    edits: list[dict[str, Any]] = []
    for score, add, hard_gain in scored:
        if len(edits) >= int(max_edits) or not drops:
            break
        drop = drops.pop(0)
        if drop in selected:
            selected.remove(drop)
        selected.add(add)
        edits.append({"drop_id": int(drop), "add_id": int(add), "score": float(score)})

    return {
        "selected_ids": sorted(selected),
        "edits": edits,
        "metadata": {
            "selected_recipe": "lsf_v7_failure_aware",
            "source_count": int(len(source_ids)),
            "candidate_count": int(len(candidate_ids)),
            "edit_count": int(len(edits)),
            "protected_query_count": int(len(protected_queries)),
            "protected_source_count": int(len(protected_sources)),
            "hard_query_count": int(len(hard_queries)),
            "rejected_protected_regression_count": int(rejected_protected),
            "paper_safe_scope": "train_or_selfmap_failure_aware_selection",
        },
    }
