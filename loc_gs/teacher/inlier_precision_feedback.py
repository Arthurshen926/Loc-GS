from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split
from loc_gs.teacher.distillation_artifact import SolverFeedbackRow


@dataclass(frozen=True)
class InlierPrecisionFeedbackConfig:
    selected_correct_ratio_floor: float = 0.10
    inlier_correct_ratio_floor: float = 0.50
    target_median_te_cm: float = 10.0
    selected_gap_gain: float = 1.0
    inlier_gap_gain: float = 2.0
    post_pnp_harm_gain: float = 1.0
    max_distill_weight: float = 4.0


@dataclass(frozen=True)
class InlierPrecisionFeedbackRow:
    scene: str
    split_name: str
    query_id: str
    query_role: str
    selected_set_hard: bool
    inlier_precision_hard: bool
    post_pnp_rescore_harm: bool
    selected_geometric_correct_ratio: float | None
    inlier_geometric_correct_ratio: float | None
    selected_gap: float
    inlier_gap: float
    set_selection_weight: float
    hard_negative_weight: float
    scorer_distill_weight: float
    te_cm: float | None = None
    set_conflict_rerank_changed_count: int = 0

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "internal_inlier_precision_feedback_row_v1",
            **asdict(self),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "InlierPrecisionFeedbackRow":
        split = reject_test_split(str(payload.get("split_name", "unknown")), purpose="inlier precision feedback rows")
        query_id = str(payload.get("query_id") or "")
        if not query_id:
            raise ValueError("inlier precision feedback row is missing query_id")
        return cls(
            scene=str(payload.get("scene", "unknown")),
            split_name=split,
            query_id=query_id,
            query_role=str(payload.get("query_role", "stable")),
            selected_set_hard=bool(payload.get("selected_set_hard", False)),
            inlier_precision_hard=bool(payload.get("inlier_precision_hard", False)),
            post_pnp_rescore_harm=bool(payload.get("post_pnp_rescore_harm", False)),
            selected_geometric_correct_ratio=_maybe_float(payload.get("selected_geometric_correct_ratio")),
            inlier_geometric_correct_ratio=_maybe_float(payload.get("inlier_geometric_correct_ratio")),
            selected_gap=float(payload.get("selected_gap", 0.0)),
            inlier_gap=float(payload.get("inlier_gap", 0.0)),
            set_selection_weight=float(payload.get("set_selection_weight", 1.0)),
            hard_negative_weight=float(payload.get("hard_negative_weight", 1.0)),
            scorer_distill_weight=float(payload.get("scorer_distill_weight", 0.0)),
            te_cm=_maybe_float(payload.get("te_cm")),
            set_conflict_rerank_changed_count=_as_int(payload.get("set_conflict_rerank_changed_count")),
        )


def build_inlier_precision_feedback(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    scene: str,
    split_name: str,
    cfg: InlierPrecisionFeedbackConfig | None = None,
) -> dict[str, object]:
    if cfg is None:
        cfg = InlierPrecisionFeedbackConfig()
    split = reject_test_split(split_name, purpose="internal inlier precision feedback")
    feedback_rows = [
        _build_row(row, scene=str(scene), split_name=split, cfg=cfg)
        for row in rows
    ]
    hard_rows = [row for row in feedback_rows if row.query_role != "stable"]
    set_rows = [row for row in feedback_rows if row.selected_set_hard]
    inlier_rows = [row for row in feedback_rows if row.inlier_precision_hard]
    post_harm_rows = [row for row in feedback_rows if row.post_pnp_rescore_harm]
    median_te = _maybe_float(metrics.get("median_te_cm"))
    target = float(cfg.target_median_te_cm)
    weights = [float(row.scorer_distill_weight) for row in feedback_rows]
    return {
        "schema_version": "internal_inlier_precision_feedback_v1",
        "scene": str(scene),
        "split_name": split,
        "query_count": int(len(feedback_rows)),
        "success_count": _as_int(metrics.get("success_count")),
        "median_te_cm": median_te,
        "median_re_deg": _maybe_float(metrics.get("median_re_deg")),
        "target_median_te_cm": target,
        "target_gap_cm": None if median_te is None else float(median_te - target),
        "hard_query_count": int(len(hard_rows)),
        "set_selection_hard_count": int(len(set_rows)),
        "inlier_precision_hard_count": int(len(inlier_rows)),
        "post_pnp_rescore_harm_count": int(len(post_harm_rows)),
        "mean_scorer_distill_weight": float(mean(weights)) if weights else 0.0,
        "max_scorer_distill_weight": float(max(weights)) if weights else 0.0,
        "student_consumers": [
            "candidate_mlp_scorer",
            "correspondence_scorer",
            "landmark_selector",
            "conflict_graph",
        ],
        "recommendation": _recommendation(
            set_count=len(set_rows),
            inlier_count=len(inlier_rows),
            post_harm_count=len(post_harm_rows),
        ),
        "thresholds": asdict(cfg),
        "per_query": [row.to_json_dict() for row in feedback_rows],
    }


def load_inlier_precision_feedback_rows(path: str | Path) -> list[InlierPrecisionFeedbackRow]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise ValueError(f"inlier precision feedback JSON must contain an object: {path}")
        rows = payload.get("per_query", [])
        if not isinstance(rows, list):
            raise ValueError(f"inlier precision feedback per_query must be a list: {path}")
        return [InlierPrecisionFeedbackRow.from_json_dict(row) for row in rows if isinstance(row, Mapping)]
    out: list[InlierPrecisionFeedbackRow] = []
    for line_number, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"inlier precision feedback line {line_number} must be an object")
        out.append(InlierPrecisionFeedbackRow.from_json_dict(payload))
    return out


def apply_inlier_precision_feedback_to_solver_rows(
    solver_rows: Sequence[SolverFeedbackRow],
    feedback_rows: Sequence[InlierPrecisionFeedbackRow],
    *,
    weight_scale: float = 1.0,
    max_distill_weight: float = 4.0,
) -> tuple[list[SolverFeedbackRow], dict[str, object]]:
    by_query = {row.query_id: row for row in solver_rows}
    matched = 0
    added = 0
    boosted = 0
    for feedback in feedback_rows:
        reject_test_split(feedback.split_name, purpose="merge inlier precision feedback")
        boost = max(0.0, float(feedback.scorer_distill_weight)) * float(weight_scale)
        if boost <= 0.0:
            continue
        existing = by_query.get(feedback.query_id)
        if existing is None:
            by_query[feedback.query_id] = SolverFeedbackRow(
                scene=feedback.scene,
                split_name=feedback.split_name,
                query_id=feedback.query_id,
                dense_helped=False,
                distill_weight=min(float(max_distill_weight), boost),
            )
            added += 1
            continue
        matched += 1
        new_weight = min(float(max_distill_weight), float(existing.distill_weight) + boost)
        if new_weight > float(existing.distill_weight):
            boosted += 1
        by_query[feedback.query_id] = SolverFeedbackRow(
            scene=existing.scene,
            split_name=existing.split_name,
            query_id=existing.query_id,
            dense_helped=existing.dense_helped,
            distill_weight=float(new_weight),
        )
    merged = [by_query[row.query_id] for row in solver_rows]
    existing_ids = {row.query_id for row in solver_rows}
    merged.extend(row for query_id, row in sorted(by_query.items()) if query_id not in existing_ids)
    return merged, {
        "schema_version": "internal_inlier_precision_feedback_merge_summary_v1",
        "input_solver_feedback_count": int(len(solver_rows)),
        "input_inlier_precision_feedback_count": int(len(feedback_rows)),
        "matched_solver_feedback_count": int(matched),
        "added_solver_feedback_count": int(added),
        "boosted_solver_feedback_count": int(boosted),
        "weight_scale": float(weight_scale),
        "max_distill_weight": float(max_distill_weight),
    }


def _build_row(
    row: Mapping[str, Any],
    *,
    scene: str,
    split_name: str,
    cfg: InlierPrecisionFeedbackConfig,
) -> InlierPrecisionFeedbackRow:
    selected = dict(row.get("selected_set_diagnostics", {}) or {})
    inlier = dict(row.get("inlier_set_diagnostics", {}) or {})
    selected_ratio = _maybe_float(selected.get("selected_geometric_correct_ratio"))
    inlier_ratio = _maybe_float(inlier.get("inlier_geometric_correct_ratio"))
    selected_gap = _ratio_gap(selected_ratio, cfg.selected_correct_ratio_floor)
    inlier_gap = _ratio_gap(inlier_ratio, cfg.inlier_correct_ratio_floor)
    selected_hard = bool(selected_gap > 0.0)
    inlier_hard = bool(inlier_gap > 0.0)
    post_harm = _post_pnp_harm(row)
    scorer_weight = min(
        float(cfg.max_distill_weight),
        float(cfg.selected_gap_gain) * selected_gap
        + float(cfg.inlier_gap_gain) * inlier_gap
        + (float(cfg.post_pnp_harm_gain) if post_harm else 0.0),
    )
    set_weight = min(float(cfg.max_distill_weight), 1.0 + selected_gap + 0.5 * inlier_gap)
    hard_weight = min(float(cfg.max_distill_weight), 1.0 + selected_gap + inlier_gap + (1.0 if post_harm else 0.0))
    return InlierPrecisionFeedbackRow(
        scene=scene,
        split_name=split_name,
        query_id=str(row.get("query_id", "")),
        query_role=_query_role(selected_hard=selected_hard, inlier_hard=inlier_hard, post_harm=post_harm),
        selected_set_hard=selected_hard,
        inlier_precision_hard=inlier_hard,
        post_pnp_rescore_harm=post_harm,
        selected_geometric_correct_ratio=selected_ratio,
        inlier_geometric_correct_ratio=inlier_ratio,
        selected_gap=float(selected_gap),
        inlier_gap=float(inlier_gap),
        set_selection_weight=float(set_weight),
        hard_negative_weight=float(hard_weight),
        scorer_distill_weight=float(scorer_weight),
        te_cm=_maybe_float(row.get("te_cm")),
        set_conflict_rerank_changed_count=_as_int(row.get("set_conflict_rerank_changed_count")),
    )


def _ratio_gap(value: float | None, floor: float) -> float:
    if value is None:
        return 0.0
    denom = max(1.0e-9, float(floor))
    return float(max(0.0, float(floor) - float(value)) / denom)


def _post_pnp_harm(row: Mapping[str, Any]) -> bool:
    changed = _as_int(row.get("post_pnp_rescore_changed_count"))
    corrected = _as_int(row.get("post_pnp_rescore_corrected_count"))
    worsened = _as_int(row.get("post_pnp_rescore_worsened_count"))
    delta = _as_int(row.get("post_pnp_rescore_correct_delta"))
    return bool(changed > 0 and (delta < 0 or worsened > corrected))


def _query_role(*, selected_hard: bool, inlier_hard: bool, post_harm: bool) -> str:
    if selected_hard and inlier_hard:
        return "set_and_inlier_precision_hard"
    if inlier_hard:
        return "inlier_precision_hard"
    if selected_hard:
        return "set_selection_hard"
    if post_harm:
        return "post_pnp_rescore_harm"
    return "stable"


def _recommendation(*, set_count: int, inlier_count: int, post_harm_count: int) -> str:
    if set_count or inlier_count:
        return "boost_hard_negative_and_set_level_student_weights"
    if post_harm_count:
        return "suppress_post_pnp_rescore_for_hard_queries"
    return "monitor_no_hard_inlier_precision_queries"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
