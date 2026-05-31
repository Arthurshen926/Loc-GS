from __future__ import annotations

from typing import Any, Mapping, Sequence


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def compute_dense_damage_risk(row: Mapping[str, Any]) -> dict[str, Any]:
    """Score native dense damage risk from observable sparse/dense diagnostics only."""

    sparse_inliers = _float(row, "sparse_inlier_count", 0.0)
    sparse_confidence = _clip01((sparse_inliers - 32.0) / 96.0)

    sparse_anchor = _float(row, "sparse_pose_anchor_median_px", 0.0)
    dense_anchor = _float(row, "base_dense_pose_anchor_median_px", sparse_anchor)
    anchor_conflict = _clip01((dense_anchor - sparse_anchor - 0.5) / 6.0)

    translation_delta = _float(row, "base_dense_vs_sparse_translation_delta_m", 0.0)
    rotation_delta = _float(row, "base_dense_vs_sparse_rotation_delta_deg", 0.0)
    dense_p90 = _float(row, "base_dense_pose_p90_reprojection_error_px", 0.0)
    translation_risk = _clip01((translation_delta - 0.05) / 0.45)
    rotation_risk = _clip01((rotation_delta - 0.5) / 5.0)
    reproj_risk = _clip01((dense_p90 - 4.0) / 12.0)
    dense_update_risk = max(translation_risk, rotation_risk, reproj_risk)

    risk = sparse_confidence * max(anchor_conflict, dense_update_risk) * dense_update_risk
    return {
        "schema": "loc_gs_apd_dense_damage_risk_v1",
        "uses_gt": False,
        "risk": float(_clip01(risk)),
        "components": {
            "sparse_confidence": float(sparse_confidence),
            "anchor_conflict": float(anchor_conflict),
            "dense_update_risk": float(dense_update_risk),
            "translation_risk": float(translation_risk),
            "rotation_risk": float(rotation_risk),
            "reprojection_risk": float(reproj_risk),
        },
    }


def label_dense_damage_from_gt(
    row: Mapping[str, Any],
    *,
    sparse_good_max_te_cm: float = 30.0,
    dense_worsen_min_cm: float = 20.0,
) -> bool:
    """Validation-only label: sparse is reasonable and native dense worsens it."""

    sparse_te = _float(row, "sparse_te_cm", float("inf"))
    dense_te = _float(row, "base_dense_te_cm", float("inf"))
    return bool(sparse_te <= float(sparse_good_max_te_cm) and dense_te - sparse_te >= float(dense_worsen_min_cm))


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = [s for y, s in zip(labels, scores) if y]
    negatives = [s for y, s in zip(labels, scores) if not y]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total)


def evaluate_dense_damage_risk(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [label_dense_damage_from_gt(row) for row in rows]
    unsorted = [(compute_dense_damage_risk(row)["risk"], label, row) for row, label in zip(rows, labels)]
    scored = list(unsorted)
    scored.sort(key=lambda item: item[0], reverse=True)
    positive_count = int(sum(labels))
    top_k = max(1, positive_count) if scored else 0
    top = scored[:top_k]
    true_positive_top = int(sum(1 for _score, label, _row in top if label))
    precision = float(true_positive_top / top_k) if top_k else 0.0
    recall = float(true_positive_top / positive_count) if positive_count else 0.0
    return {
        "schema": "loc_gs_apd_dense_damage_risk_eval_v1",
        "uses_gt_for_scoring": False,
        "uses_gt_for_validation_labels": True,
        "query_count": int(len(rows)),
        "positive_count": int(positive_count),
        "auc": _auc([label for _score, label, _row in unsorted], [score for score, _label, _row in unsorted]),
        "precision_at_positive_count": precision,
        "recall_at_positive_count": recall,
        "top_scores": [
            {
                "risk": float(score),
                "label": bool(label),
                "scene": row.get("scene"),
                "image_name": row.get("image_name"),
            }
            for score, label, row in scored[: min(20, len(scored))]
        ],
    }
