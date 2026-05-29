from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def _score_match(row: Mapping[str, Any]) -> float:
    score = float(row.get("score", 0.0) or 0.0)
    inlier = float(row.get("inlier_prior", row.get("inliers", 0.0)) or 0.0)
    logdet = float(row.get("logdet_h", row.get("logdet_H", 0.0)) or 0.0)
    dense = float(row.get("dense_verifier", row.get("dense_score", 0.0)) or 0.0)
    return float(score + 0.10 * inlier + 0.25 * logdet + dense)


def build_multihypothesis_diagnostic_plan(
    *,
    matches: Sequence[Mapping[str, Any]],
    diagnostic: bool,
    max_hypotheses: int = 8,
) -> dict[str, Any]:
    """Build diversity-aware PnP hypothesis groups as a diagnostic-only artifact."""

    if not bool(diagnostic):
        raise ValueError("multi-hypothesis PnP is diagnostic-only and requires diagnostic=True")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in matches:
        grouped[str(row.get("group", row.get("bank", row.get("spatial_group", "default"))))].append(row)
    hypotheses: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        score = sum(_score_match(row) for row in rows)
        hypotheses.append(
            {
                "group": group,
                "match_count": int(len(rows)),
                "score": float(score),
                "solver_backend": "same_backend_as_stdloc_diagnostic",
            }
        )
    hypotheses.sort(key=lambda item: (-float(item["score"]), str(item["group"])))
    selected = hypotheses[: max(1, int(max_hypotheses))]
    return {
        "hypotheses": selected,
        "metadata": {
            "recipe": "lsf_v12_multihyp_diagnostic",
            "paper_safe_role": "diagnostic_ablation_only",
            "main_method_allowed": False,
            "oracle_ordering": False,
            "test_tuning_allowed": False,
            "input_match_count": int(sum(len(rows) for rows in grouped.values())),
            "hypothesis_count": int(len(selected)),
        },
    }
