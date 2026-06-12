from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateRerankConfig:
    prefix_fraction: float = 0.0
    solver_weight: float = 1.0
    native_weight: float = 1.0


def _score(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def rerank_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    cfg: CandidateRerankConfig | None = None,
) -> list[dict[str, Any]]:
    if cfg is None:
        cfg = CandidateRerankConfig()
    if not rows:
        return []
    ordered = sorted(enumerate(rows), key=lambda item: (-_score(item[1], "native_score"), item[0]))
    prefix_fraction = max(0.0, min(1.0, float(cfg.prefix_fraction)))
    prefix_count = int(round(len(ordered) * prefix_fraction))
    prefix = [dict(row) for _idx, row in ordered[:prefix_count]]
    tail = sorted(
        ordered[prefix_count:],
        key=lambda item: (
            -(
                float(cfg.native_weight) * _score(item[1], "native_score")
                + float(cfg.solver_weight) * _score(item[1], "solver_score")
            ),
            item[0],
        ),
    )
    return prefix + [dict(row) for _idx, row in tail]


def summarize_candidate_availability(candidate_rows_by_query: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    query_count = len(candidate_rows_by_query)
    top1_correct = 0
    topk_available = 0
    for rows in candidate_rows_by_query:
        if rows and bool(rows[0].get("geometric_correct", False)):
            top1_correct += 1
        if any(bool(row.get("geometric_correct", False)) for row in rows):
            topk_available += 1
    return {
        "query_count": int(query_count),
        "top1_correct": int(top1_correct),
        "topk_available": int(topk_available),
        "oracle_gap": int(topk_available - top1_correct),
    }
