#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAMBRIDGE_SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)

RECALL_THRESHOLDS = {
    "R50cm5deg": (50.0, 5.0),
    "R15cm5deg": (15.0, 5.0),
    "R10cm5deg": (10.0, 5.0),
    "R5cm5deg": (5.0, 5.0),
    "R2cm2deg": (2.0, 2.0),
}

INLIER_BINS = (
    ("inliers_low", -math.inf, 128.0),
    ("inliers_mid", 128.0, 512.0),
    ("inliers_high", 512.0, math.inf),
)

DEFAULT_STUMP_GATE_FEATURES = (
    "log_sparse_inliers",
    "sparse_pose_inlier_2px",
    "sparse_pose_inlier_5px",
    "sparse_pose_inlier_8px",
    "sparse_pose_reproj_median_px",
    "sparse_pose_inlier8_pose_info_logdet",
    "sparse_pose_inlier8_pose_info_min_eig",
    "sparse_pose_inlier8_image_cov_logdet",
    "sparse_pose_inlier8_xyz_cov_logdet",
    "sparse_pose_match_score_median",
    "sparse_pose_top2_margin_median",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = min(1.0, max(0.0, float(q))) * float(len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - float(low)
    return float((1.0 - weight) * ordered[low] + weight * ordered[high])


def _tail_cvar(values: Sequence[float], fraction: float = 0.10) -> float | None:
    if not values:
        return None
    ordered = sorted((float(v) for v in values), reverse=True)
    count = max(1, int(math.ceil(len(ordered) * max(0.0, float(fraction)))))
    return float(sum(ordered[:count]) / count)


def _stage_metric(row: Mapping[str, Any], stage: str, metric: str) -> float | None:
    candidates = {
        "te": (f"{stage}_te", f"{stage}_TE", f"{stage}_te_cm"),
        "re": (f"{stage}_ae", f"{stage}_AE", f"{stage}_re_deg"),
        "inliers": (f"{stage}_inliers",),
    }[metric]
    for key in candidates:
        value = _safe_float(row.get(key))
        if value is not None:
            return value

    nested = row.get(stage)
    if isinstance(nested, Mapping):
        keys = {"te": ("te", "TE", "te_cm"), "re": ("ae", "AE", "re_deg"), "inliers": ("inliers",)}[metric]
        for key in keys:
            value = _safe_float(nested.get(key))
            if value is not None:
                return value
    if metric == "inliers" and isinstance(nested, list) and nested:
        last = nested[-1]
        if isinstance(last, Mapping):
            return _safe_float(last.get("inliers"))
    return None


def _query_id(row: Mapping[str, Any], index: int) -> str:
    image_name = row.get("image_name")
    if image_name:
        return str(image_name)
    return f"query_{index:06d}"


def load_sparse_rows(run_dir: Path, *, stage: str = "sparse") -> list[dict[str, Any]]:
    raw = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), list):
        raw = raw["results"]
    if not isinstance(raw, list):
        raise ValueError(f"results.json must contain a list: {run_dir / 'results.json'}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        te = _stage_metric(item, stage, "te")
        re = _stage_metric(item, stage, "re")
        rows.append(
            {
                "query_id": _query_id(item, index),
                "query_index": index,
                "image_name": item.get("image_name", _query_id(item, index)),
                f"{stage}_te_cm": te,
                f"{stage}_re_deg": re,
                f"{stage}_inliers": _stage_metric(item, stage, "inliers"),
                "sparse_confidence": item.get("sparse_confidence") if isinstance(item.get("sparse_confidence"), Mapping) else {},
                "localized": bool(item.get("localized", te is not None)),
            }
        )
    return rows


def _recall(rows: Sequence[Mapping[str, Any]], *, stage: str, max_te_cm: float, max_re_deg: float) -> float | None:
    valid = [
        (row.get(f"{stage}_te_cm"), row.get(f"{stage}_re_deg"))
        for row in rows
        if row.get(f"{stage}_te_cm") is not None and row.get(f"{stage}_re_deg") is not None
    ]
    if not valid:
        return None
    hits = sum(1 for te, re in valid if float(te) <= max_te_cm and float(re) <= max_re_deg)
    return float(hits / len(valid))


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, stage: str = "sparse") -> dict[str, Any]:
    tes = [float(row[f"{stage}_te_cm"]) for row in rows if row.get(f"{stage}_te_cm") is not None]
    res = [float(row[f"{stage}_re_deg"]) for row in rows if row.get(f"{stage}_re_deg") is not None]
    inliers = [float(row[f"{stage}_inliers"]) for row in rows if row.get(f"{stage}_inliers") is not None]
    out: dict[str, Any] = {
        "query_count": len(rows),
        "valid_count": len(tes),
        "median_te_cm": _median(tes),
        "median_re_deg": _median(res),
        "p90_te_cm": _percentile(tes, 0.90),
        "p95_te_cm": _percentile(tes, 0.95),
        "cvar10_te_cm": _tail_cvar(tes, 0.10),
        "severe_rate_1m": float(sum(1 for value in tes if value > 100.0) / len(tes)) if tes else None,
        "catastrophic_rate_5m": float(sum(1 for value in tes if value > 500.0) / len(tes)) if tes else None,
        "avg_inliers": _mean(inliers),
    }
    for name, (te_thr, re_thr) in RECALL_THRESHOLDS.items():
        out[name] = _recall(rows, stage=stage, max_te_cm=te_thr, max_re_deg=re_thr)
    return out


def _inlier_bin(inliers: float | None) -> str:
    if inliers is None:
        return "inliers_unknown"
    for name, low, high in INLIER_BINS:
        if float(inliers) >= low and float(inliers) < high:
            return name
    return "inliers_unknown"


def _paired_query_rows(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    candidate_by_id = {str(row["query_id"]): row for row in candidate_rows}
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for base in base_rows:
        candidate = candidate_by_id.get(str(base["query_id"]))
        if candidate is not None:
            paired.append((base, candidate))
    return paired


def _choose_oracle_row(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    base_te = _safe_float(base.get(f"{stage}_te_cm"))
    cand_te = _safe_float(candidate.get(f"{stage}_te_cm"))
    choose_candidate = cand_te is not None and (base_te is None or cand_te < base_te)
    chosen = candidate if choose_candidate else base
    return {
        "query_id": base["query_id"],
        "query_index": base.get("query_index"),
        "image_name": base.get("image_name"),
        f"{stage}_te_cm": _safe_float(chosen.get(f"{stage}_te_cm")),
        f"{stage}_re_deg": _safe_float(chosen.get(f"{stage}_re_deg")),
        f"{stage}_inliers": _safe_float(chosen.get(f"{stage}_inliers")),
        "oracle_source": "candidate" if choose_candidate else "base",
    }


def _paired_report(
    *,
    scene: str,
    candidate_name: str,
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    stage: str,
    improvement_margin_cm: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paired = _paired_query_rows(base_rows, candidate_rows)
    oracle_rows = [_choose_oracle_row(base, candidate, stage=stage) for base, candidate in paired]
    helped = harmed = unchanged = raw_better = raw_worse = 0
    regression_20 = regression_50 = improvement_20 = 0
    case_rows: list[dict[str, Any]] = []
    bin_acc: dict[str, list[dict[str, Any]]] = {}
    for base, candidate in paired:
        base_te = _safe_float(base.get(f"{stage}_te_cm"))
        candidate_te = _safe_float(candidate.get(f"{stage}_te_cm"))
        delta = None if base_te is None or candidate_te is None else candidate_te - base_te
        label = "unknown"
        if delta is not None:
            raw_better += int(delta < 0.0)
            raw_worse += int(delta > 0.0)
            if delta <= -float(improvement_margin_cm):
                helped += 1
                label = "helped"
            elif delta >= float(improvement_margin_cm):
                harmed += 1
                label = "harmed"
            else:
                unchanged += 1
                label = "unchanged"
            regression_20 += int(delta >= 20.0)
            regression_50 += int(delta >= 50.0)
            improvement_20 += int(delta <= -20.0)

        bin_name = _inlier_bin(_safe_float(base.get(f"{stage}_inliers")))
        row = {
            "scene": scene,
            "candidate": candidate_name,
            "query_id": base["query_id"],
            "image_name": base.get("image_name"),
            "base_te_cm": base_te,
            "candidate_te_cm": candidate_te,
            "delta_te_cm": delta,
            "base_re_deg": _safe_float(base.get(f"{stage}_re_deg")),
            "candidate_re_deg": _safe_float(candidate.get(f"{stage}_re_deg")),
            "base_inliers": _safe_float(base.get(f"{stage}_inliers")),
            "candidate_inliers": _safe_float(candidate.get(f"{stage}_inliers")),
            "confidence_bin": bin_name,
            "label": label,
        }
        case_rows.append(row)
        bin_acc.setdefault(bin_name, []).append(row)

    paired_summary = {
        "query_count": len(paired),
        "helped_count": int(helped),
        "harmed_count": int(harmed),
        "unchanged_count": int(unchanged),
        "raw_better_count": int(raw_better),
        "raw_worse_count": int(raw_worse),
        "oracle_selected_candidate_count": int(sum(1 for row in oracle_rows if row["oracle_source"] == "candidate")),
        "regression_20cm_count": int(regression_20),
        "regression_50cm_count": int(regression_50),
        "improvement_20cm_count": int(improvement_20),
        "mean_delta_te_cm": _mean(row.get("delta_te_cm") for row in case_rows),
    }
    confidence_bins: list[dict[str, Any]] = []
    for bin_name in [name for name, _, _ in INLIER_BINS] + ["inliers_unknown"]:
        rows = bin_acc.get(bin_name, [])
        if not rows:
            continue
        confidence_bins.append(
            {
                "bin": bin_name,
                "query_count": len(rows),
                "helped_count": sum(1 for row in rows if row["label"] == "helped"),
                "harmed_count": sum(1 for row in rows if row["label"] == "harmed"),
                "unchanged_count": sum(1 for row in rows if row["label"] == "unchanged"),
                "mean_delta_te_cm": _mean(row.get("delta_te_cm") for row in rows),
            }
        )
    return paired_summary, confidence_bins, oracle_rows, case_rows


def _learn_active_inlier_bins(
    train_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    stage: str,
    improvement_margin_cm: float,
    min_train_queries: int,
    min_help_minus_harm: int,
    min_median_gain_cm: float,
    max_regression_excess: int,
) -> dict[str, dict[str, Any]]:
    by_bin: dict[str, list[float]] = {}
    for base, candidate in train_pairs:
        base_te = _safe_float(base.get(f"{stage}_te_cm"))
        candidate_te = _safe_float(candidate.get(f"{stage}_te_cm"))
        if base_te is None or candidate_te is None:
            continue
        bin_name = _inlier_bin(_safe_float(base.get(f"{stage}_inliers")))
        by_bin.setdefault(bin_name, []).append(candidate_te - base_te)

    policies: dict[str, dict[str, Any]] = {}
    for bin_name in [name for name, _, _ in INLIER_BINS] + ["inliers_unknown"]:
        deltas = by_bin.get(bin_name, [])
        helped = sum(1 for delta in deltas if delta <= -float(improvement_margin_cm))
        harmed = sum(1 for delta in deltas if delta >= float(improvement_margin_cm))
        improvement_20 = sum(1 for delta in deltas if delta <= -20.0)
        regression_20 = sum(1 for delta in deltas if delta >= 20.0)
        median_delta = _median(deltas)
        activate = (
            len(deltas) >= int(min_train_queries)
            and helped - harmed >= int(min_help_minus_harm)
            and median_delta is not None
            and median_delta <= -float(min_median_gain_cm)
            and regression_20 - improvement_20 <= int(max_regression_excess)
        )
        policies[bin_name] = {
            "activate": bool(activate),
            "train_count": len(deltas),
            "train_helped_count": int(helped),
            "train_harmed_count": int(harmed),
            "train_improvement_20cm_count": int(improvement_20),
            "train_regression_20cm_count": int(regression_20),
            "train_median_delta_te_cm": median_delta,
            "train_mean_delta_te_cm": _mean(deltas),
        }
    return policies


def build_inlier_bin_gate_cv(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    stage: str = "sparse",
    folds: int = 5,
    improvement_margin_cm: float = 1.0,
    min_train_queries: int = 30,
    min_help_minus_harm: int = 1,
    min_median_gain_cm: float = 0.0,
    max_regression_excess: int = 0,
) -> dict[str, Any]:
    paired = _paired_query_rows(base_rows, candidate_rows)
    if not paired:
        return {
            "summary": summarize_rows([], stage=stage),
            "paired": {"query_count": 0, "activated_count": 0},
            "folds": [],
            "decisions": [],
        }
    folds = max(2, int(folds))
    decisions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(folds):
        train_pairs = [pair for index, pair in enumerate(paired) if index % folds != fold]
        eval_pairs = [pair for index, pair in enumerate(paired) if index % folds == fold]
        policies = _learn_active_inlier_bins(
            train_pairs,
            stage=stage,
            improvement_margin_cm=improvement_margin_cm,
            min_train_queries=min_train_queries,
            min_help_minus_harm=min_help_minus_harm,
            min_median_gain_cm=min_median_gain_cm,
            max_regression_excess=max_regression_excess,
        )
        fold_reports.append({"fold": fold, "train_count": len(train_pairs), "eval_count": len(eval_pairs), "policies": policies})
        for base, candidate in eval_pairs:
            bin_name = _inlier_bin(_safe_float(base.get(f"{stage}_inliers")))
            activate = bool(policies.get(bin_name, {}).get("activate", False))
            chosen = candidate if activate else base
            base_te = _safe_float(base.get(f"{stage}_te_cm"))
            chosen_te = _safe_float(chosen.get(f"{stage}_te_cm"))
            delta = None if base_te is None or chosen_te is None else chosen_te - base_te
            if delta is None:
                label = "unknown"
            elif delta <= -float(improvement_margin_cm):
                label = "helped"
            elif delta >= float(improvement_margin_cm):
                label = "harmed"
            else:
                label = "unchanged"
            decisions.append(
                {
                    "query_id": base["query_id"],
                    "query_index": base.get("query_index"),
                    "image_name": base.get("image_name"),
                    "fold": fold,
                    "confidence_bin": bin_name,
                    "activated": activate,
                    "source": "candidate" if activate else "base",
                    f"{stage}_te_cm": chosen_te,
                    f"{stage}_re_deg": _safe_float(chosen.get(f"{stage}_re_deg")),
                    f"{stage}_inliers": _safe_float(chosen.get(f"{stage}_inliers")),
                    "base_te_cm": base_te,
                    "candidate_te_cm": _safe_float(candidate.get(f"{stage}_te_cm")),
                    "delta_te_cm": delta,
                    "label": label,
                }
            )

    paired_summary = {
        "query_count": len(decisions),
        "activated_count": sum(1 for row in decisions if row["activated"]),
        "helped_count": sum(1 for row in decisions if row["label"] == "helped"),
        "harmed_count": sum(1 for row in decisions if row["label"] == "harmed"),
        "unchanged_count": sum(1 for row in decisions if row["label"] == "unchanged"),
        "regression_20cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) >= 20.0),
        "regression_50cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) >= 50.0),
        "improvement_20cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) <= -20.0),
        "mean_delta_te_cm": _mean(row.get("delta_te_cm") for row in decisions),
    }
    return {
        "summary": summarize_rows(decisions, stage=stage),
        "paired": paired_summary,
        "folds": fold_reports,
        "decisions": decisions,
        "policy": {
            "type": "cross_validated_inlier_bin_gate",
            "folds": folds,
            "min_train_queries": min_train_queries,
            "min_help_minus_harm": min_help_minus_harm,
            "min_median_gain_cm": min_median_gain_cm,
            "max_regression_excess": max_regression_excess,
            "uses_gt_for_fold_labels": True,
            "uses_gt_at_inference": False,
        },
    }


def _feature_value(row: Mapping[str, Any], feature: str, *, stage: str) -> float | None:
    if feature == "log_sparse_inliers":
        value = _safe_float(row.get(f"{stage}_inliers"))
        return math.log1p(value) if value is not None else None
    value = _safe_float(row.get(feature))
    if value is not None:
        return value
    confidence = row.get("sparse_confidence")
    if isinstance(confidence, Mapping):
        return _safe_float(confidence.get(feature))
    return None


def _candidate_thresholds(values: Sequence[float]) -> list[float]:
    finite = sorted({float(v) for v in values if math.isfinite(float(v))})
    if not finite:
        return []
    if len(finite) == 1:
        return [finite[0]]
    if len(finite) <= 64:
        return [0.5 * (finite[i] + finite[i + 1]) for i in range(len(finite) - 1)]
    return [_percentile(finite, q / 10.0) for q in range(1, 10) if _percentile(finite, q / 10.0) is not None]


def _fit_feature_stump(
    train_examples: Sequence[dict[str, Any]],
    *,
    features: Sequence[str],
    improvement_margin_cm: float,
    min_train_queries: int,
    min_train_activated: int,
    min_score: float,
    min_median_gain_cm: float,
    regression_50_penalty: float,
    stump_max_train_regression_20: int | None,
    stump_max_train_regression_50: int | None,
) -> dict[str, Any]:
    best: dict[str, Any] = {"activate": False, "score": 0.0}
    if len(train_examples) < int(min_train_queries):
        return best
    for feature in features:
        values = [example["features"].get(feature) for example in train_examples]
        values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        for threshold in _candidate_thresholds(values):
            for direction in ("le", "ge"):
                active = []
                for example in train_examples:
                    value = example["features"].get(feature)
                    if value is None or not math.isfinite(float(value)):
                        continue
                    is_active = float(value) <= float(threshold) if direction == "le" else float(value) >= float(threshold)
                    if is_active:
                        active.append(example)
                if len(active) < int(min_train_activated):
                    continue
                deltas = [float(example["delta_te_cm"]) for example in active if example.get("delta_te_cm") is not None]
                if not deltas:
                    continue
                helped = sum(1 for delta in deltas if delta <= -float(improvement_margin_cm))
                harmed = sum(1 for delta in deltas if delta >= float(improvement_margin_cm))
                regression_20 = sum(1 for delta in deltas if delta >= 20.0)
                regression_50 = sum(1 for delta in deltas if delta >= 50.0)
                improvement_20 = sum(1 for delta in deltas if delta <= -20.0)
                median_delta = _median(deltas)
                if stump_max_train_regression_20 is not None and regression_20 > int(stump_max_train_regression_20):
                    continue
                if stump_max_train_regression_50 is not None and regression_50 > int(stump_max_train_regression_50):
                    continue
                score = float(helped - harmed + improvement_20 - 2 * regression_20 - float(regression_50_penalty) * regression_50)
                if median_delta is None or median_delta > -float(min_median_gain_cm):
                    continue
                if score <= float(min_score):
                    continue
                if score > float(best.get("score", -math.inf)):
                    best = {
                        "activate": True,
                        "feature": feature,
                        "threshold": float(threshold),
                        "direction": direction,
                        "score": score,
                        "train_activated_count": len(active),
                        "train_helped_count": int(helped),
                        "train_harmed_count": int(harmed),
                        "train_improvement_20cm_count": int(improvement_20),
                        "train_regression_20cm_count": int(regression_20),
                        "train_regression_50cm_count": int(regression_50),
                        "train_median_delta_te_cm": median_delta,
                        "train_mean_delta_te_cm": _mean(deltas),
                    }
    return best


def _condition_from_stump(stump: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature": str(stump["feature"]),
        "threshold": float(stump["threshold"]),
        "direction": str(stump["direction"]),
    }


def _condition_matches(features: Mapping[str, Any], condition: Mapping[str, Any]) -> tuple[bool, float | None]:
    value = _safe_float(features.get(str(condition["feature"])))
    if value is None:
        return False, None
    threshold = float(condition["threshold"])
    if str(condition["direction"]) == "le":
        return value <= threshold, value
    return value >= threshold, value


def _conditions_match(features: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, float | None]]:
    values: dict[str, float | None] = {}
    for condition in conditions:
        matched, value = _condition_matches(features, condition)
        values[str(condition["feature"])] = value
        if not matched:
            return False, values
    return True, values


def _score_active_deltas(
    deltas: Sequence[float],
    *,
    improvement_margin_cm: float,
    min_train_activated: int,
    min_score: float,
    min_median_gain_cm: float,
    regression_50_penalty: float,
    stump_max_train_regression_20: int | None,
    stump_max_train_regression_50: int | None,
) -> dict[str, Any]:
    helped = sum(1 for delta in deltas if delta <= -float(improvement_margin_cm))
    harmed = sum(1 for delta in deltas if delta >= float(improvement_margin_cm))
    regression_20 = sum(1 for delta in deltas if delta >= 20.0)
    regression_50 = sum(1 for delta in deltas if delta >= 50.0)
    improvement_20 = sum(1 for delta in deltas if delta <= -20.0)
    median_delta = _median(deltas)
    score = float(helped - harmed + improvement_20 - 2 * regression_20 - float(regression_50_penalty) * regression_50)
    activate = (
        len(deltas) >= int(min_train_activated)
        and median_delta is not None
        and median_delta <= -float(min_median_gain_cm)
        and score > float(min_score)
        and (stump_max_train_regression_20 is None or regression_20 <= int(stump_max_train_regression_20))
        and (stump_max_train_regression_50 is None or regression_50 <= int(stump_max_train_regression_50))
    )
    return {
        "activate": bool(activate),
        "score": score,
        "train_activated_count": len(deltas),
        "train_helped_count": int(helped),
        "train_harmed_count": int(harmed),
        "train_improvement_20cm_count": int(improvement_20),
        "train_regression_20cm_count": int(regression_20),
        "train_regression_50cm_count": int(regression_50),
        "train_median_delta_te_cm": median_delta,
        "train_mean_delta_te_cm": _mean(deltas),
    }


def _evaluate_rule_policy(
    train_examples: Sequence[dict[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
    *,
    improvement_margin_cm: float,
    min_train_activated: int,
    min_score: float,
    min_median_gain_cm: float,
    regression_50_penalty: float,
    stump_max_train_regression_20: int | None,
    stump_max_train_regression_50: int | None,
) -> dict[str, Any]:
    deltas = [
        float(example["delta_te_cm"])
        for example in train_examples
        if example.get("delta_te_cm") is not None and _conditions_match(example["features"], conditions)[0]
    ]
    stats = _score_active_deltas(
        deltas,
        improvement_margin_cm=improvement_margin_cm,
        min_train_activated=min_train_activated,
        min_score=min_score,
        min_median_gain_cm=min_median_gain_cm,
        regression_50_penalty=regression_50_penalty,
        stump_max_train_regression_20=stump_max_train_regression_20,
        stump_max_train_regression_50=stump_max_train_regression_50,
    )
    stats["conditions"] = [dict(condition) for condition in conditions]
    return stats


def _fit_feature_rule(
    train_examples: Sequence[dict[str, Any]],
    *,
    features: Sequence[str],
    max_depth: int,
    improvement_margin_cm: float,
    min_train_queries: int,
    min_train_activated: int,
    min_score: float,
    min_median_gain_cm: float,
    regression_50_penalty: float,
    stump_max_train_regression_20: int | None,
    stump_max_train_regression_50: int | None,
) -> dict[str, Any]:
    first = _fit_feature_stump(
        train_examples,
        features=features,
        improvement_margin_cm=improvement_margin_cm,
        min_train_queries=min_train_queries,
        min_train_activated=min_train_activated,
        min_score=min_score,
        min_median_gain_cm=min_median_gain_cm,
        regression_50_penalty=regression_50_penalty,
        stump_max_train_regression_20=stump_max_train_regression_20,
        stump_max_train_regression_50=stump_max_train_regression_50,
    )
    if not first.get("activate"):
        return {"activate": False, "score": 0.0, "conditions": []}
    best = _evaluate_rule_policy(
        train_examples,
        [_condition_from_stump(first)],
        improvement_margin_cm=improvement_margin_cm,
        min_train_activated=min_train_activated,
        min_score=min_score,
        min_median_gain_cm=min_median_gain_cm,
        regression_50_penalty=regression_50_penalty,
        stump_max_train_regression_20=stump_max_train_regression_20,
        stump_max_train_regression_50=stump_max_train_regression_50,
    )
    best["activate"] = True
    if int(max_depth) <= 1:
        return best

    active_train = [example for example in train_examples if _conditions_match(example["features"], best["conditions"])[0]]
    second = _fit_feature_stump(
        active_train,
        features=features,
        improvement_margin_cm=improvement_margin_cm,
        min_train_queries=min_train_activated,
        min_train_activated=min_train_activated,
        min_score=min_score,
        min_median_gain_cm=min_median_gain_cm,
        regression_50_penalty=regression_50_penalty,
        stump_max_train_regression_20=stump_max_train_regression_20,
        stump_max_train_regression_50=stump_max_train_regression_50,
    )
    if not second.get("activate"):
        return best
    candidate = _evaluate_rule_policy(
        train_examples,
        [*best["conditions"], _condition_from_stump(second)],
        improvement_margin_cm=improvement_margin_cm,
        min_train_activated=min_train_activated,
        min_score=min_score,
        min_median_gain_cm=min_median_gain_cm,
        regression_50_penalty=regression_50_penalty,
        stump_max_train_regression_20=stump_max_train_regression_20,
        stump_max_train_regression_50=stump_max_train_regression_50,
    )
    if candidate.get("activate") and (
        float(candidate.get("score", -math.inf)) > float(best.get("score", -math.inf))
        or (
            float(candidate.get("score", -math.inf)) == float(best.get("score", -math.inf))
            and int(candidate.get("train_harmed_count", 10**9)) < int(best.get("train_harmed_count", 10**9))
        )
    ):
        candidate["activate"] = True
        return candidate
    return best


def build_feature_stump_gate_cv(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str] = DEFAULT_STUMP_GATE_FEATURES,
    stage: str = "sparse",
    folds: int = 5,
    improvement_margin_cm: float = 1.0,
    min_train_queries: int = 30,
    min_train_activated: int = 20,
    min_score: float = 0.0,
    min_median_gain_cm: float = 0.0,
    regression_50_penalty: float = 4.0,
    stump_max_train_regression_20: int | None = None,
    stump_max_train_regression_50: int | None = 0,
) -> dict[str, Any]:
    paired = _paired_query_rows(base_rows, candidate_rows)
    examples: list[dict[str, Any]] = []
    for index, (base, candidate) in enumerate(paired):
        base_te = _safe_float(base.get(f"{stage}_te_cm"))
        candidate_te = _safe_float(candidate.get(f"{stage}_te_cm"))
        features_i = {feature: _feature_value(base, feature, stage=stage) for feature in features}
        examples.append(
            {
                "index": index,
                "base": base,
                "candidate": candidate,
                "delta_te_cm": None if base_te is None or candidate_te is None else candidate_te - base_te,
                "features": features_i,
            }
        )
    if not examples:
        return {
            "summary": summarize_rows([], stage=stage),
            "paired": {"query_count": 0, "activated_count": 0},
            "folds": [],
            "decisions": [],
        }
    folds = max(2, int(folds))
    decisions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [example for example in examples if int(example["index"]) % folds != fold]
        eval_examples = [example for example in examples if int(example["index"]) % folds == fold]
        policy = _fit_feature_stump(
            train,
            features=features,
            improvement_margin_cm=improvement_margin_cm,
            min_train_queries=min_train_queries,
            min_train_activated=min_train_activated,
            min_score=min_score,
            min_median_gain_cm=min_median_gain_cm,
            regression_50_penalty=regression_50_penalty,
            stump_max_train_regression_20=stump_max_train_regression_20,
            stump_max_train_regression_50=stump_max_train_regression_50,
        )
        fold_reports.append({"fold": fold, "train_count": len(train), "eval_count": len(eval_examples), "policy": policy})
        for example in eval_examples:
            base = example["base"]
            candidate = example["candidate"]
            activate = False
            feature_value = None
            if policy.get("activate"):
                feature = str(policy["feature"])
                feature_value = example["features"].get(feature)
                if feature_value is not None and math.isfinite(float(feature_value)):
                    if policy["direction"] == "le":
                        activate = float(feature_value) <= float(policy["threshold"])
                    else:
                        activate = float(feature_value) >= float(policy["threshold"])
            chosen = candidate if activate else base
            base_te = _safe_float(base.get(f"{stage}_te_cm"))
            chosen_te = _safe_float(chosen.get(f"{stage}_te_cm"))
            delta = None if base_te is None or chosen_te is None else chosen_te - base_te
            if delta is None:
                label = "unknown"
            elif delta <= -float(improvement_margin_cm):
                label = "helped"
            elif delta >= float(improvement_margin_cm):
                label = "harmed"
            else:
                label = "unchanged"
            decisions.append(
                {
                    "query_id": base["query_id"],
                    "query_index": base.get("query_index"),
                    "image_name": base.get("image_name"),
                    "fold": fold,
                    "activated": activate,
                    "source": "candidate" if activate else "base",
                    "policy_feature": policy.get("feature"),
                    "policy_direction": policy.get("direction"),
                    "policy_threshold": policy.get("threshold"),
                    "feature_value": feature_value,
                    f"{stage}_te_cm": chosen_te,
                    f"{stage}_re_deg": _safe_float(chosen.get(f"{stage}_re_deg")),
                    f"{stage}_inliers": _safe_float(chosen.get(f"{stage}_inliers")),
                    "base_te_cm": base_te,
                    "candidate_te_cm": _safe_float(candidate.get(f"{stage}_te_cm")),
                    "delta_te_cm": delta,
                    "label": label,
                }
            )
    paired_summary = {
        "query_count": len(decisions),
        "activated_count": sum(1 for row in decisions if row["activated"]),
        "helped_count": sum(1 for row in decisions if row["label"] == "helped"),
        "harmed_count": sum(1 for row in decisions if row["label"] == "harmed"),
        "unchanged_count": sum(1 for row in decisions if row["label"] == "unchanged"),
        "regression_20cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) >= 20.0),
        "regression_50cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) >= 50.0),
        "improvement_20cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) <= -20.0),
        "mean_delta_te_cm": _mean(row.get("delta_te_cm") for row in decisions),
    }
    return {
        "summary": summarize_rows(decisions, stage=stage),
        "paired": paired_summary,
        "folds": fold_reports,
        "decisions": decisions,
        "policy": {
            "type": "cross_validated_feature_stump_gate",
            "features": list(features),
            "folds": folds,
            "min_train_queries": min_train_queries,
            "min_train_activated": min_train_activated,
            "min_score": min_score,
            "min_median_gain_cm": min_median_gain_cm,
            "regression_50_penalty": regression_50_penalty,
            "stump_max_train_regression_20": stump_max_train_regression_20,
            "stump_max_train_regression_50": stump_max_train_regression_50,
            "uses_gt_for_fold_labels": True,
            "uses_gt_at_inference": False,
        },
    }


def build_feature_rule_gate_cv(
    base_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str] = DEFAULT_STUMP_GATE_FEATURES,
    stage: str = "sparse",
    folds: int = 5,
    max_depth: int = 2,
    improvement_margin_cm: float = 1.0,
    min_train_queries: int = 30,
    min_train_activated: int = 20,
    min_score: float = 0.0,
    min_median_gain_cm: float = 0.0,
    regression_50_penalty: float = 4.0,
    stump_max_train_regression_20: int | None = None,
    stump_max_train_regression_50: int | None = 0,
) -> dict[str, Any]:
    paired = _paired_query_rows(base_rows, candidate_rows)
    examples: list[dict[str, Any]] = []
    for index, (base, candidate) in enumerate(paired):
        base_te = _safe_float(base.get(f"{stage}_te_cm"))
        candidate_te = _safe_float(candidate.get(f"{stage}_te_cm"))
        features_i = {feature: _feature_value(base, feature, stage=stage) for feature in features}
        examples.append(
            {
                "index": index,
                "base": base,
                "candidate": candidate,
                "delta_te_cm": None if base_te is None or candidate_te is None else candidate_te - base_te,
                "features": features_i,
            }
        )
    if not examples:
        return {
            "summary": summarize_rows([], stage=stage),
            "paired": {"query_count": 0, "activated_count": 0},
            "folds": [],
            "decisions": [],
        }
    folds = max(2, int(folds))
    decisions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [example for example in examples if int(example["index"]) % folds != fold]
        eval_examples = [example for example in examples if int(example["index"]) % folds == fold]
        policy = _fit_feature_rule(
            train,
            features=features,
            max_depth=max_depth,
            improvement_margin_cm=improvement_margin_cm,
            min_train_queries=min_train_queries,
            min_train_activated=min_train_activated,
            min_score=min_score,
            min_median_gain_cm=min_median_gain_cm,
            regression_50_penalty=regression_50_penalty,
            stump_max_train_regression_20=stump_max_train_regression_20,
            stump_max_train_regression_50=stump_max_train_regression_50,
        )
        fold_reports.append({"fold": fold, "train_count": len(train), "eval_count": len(eval_examples), "policy": policy})
        for example in eval_examples:
            base = example["base"]
            candidate = example["candidate"]
            activate = False
            feature_values: dict[str, float | None] = {}
            if policy.get("activate"):
                activate, feature_values = _conditions_match(example["features"], policy.get("conditions", []))
            chosen = candidate if activate else base
            base_te = _safe_float(base.get(f"{stage}_te_cm"))
            chosen_te = _safe_float(chosen.get(f"{stage}_te_cm"))
            delta = None if base_te is None or chosen_te is None else chosen_te - base_te
            if delta is None:
                label = "unknown"
            elif delta <= -float(improvement_margin_cm):
                label = "helped"
            elif delta >= float(improvement_margin_cm):
                label = "harmed"
            else:
                label = "unchanged"
            decisions.append(
                {
                    "query_id": base["query_id"],
                    "query_index": base.get("query_index"),
                    "image_name": base.get("image_name"),
                    "fold": fold,
                    "activated": activate,
                    "source": "candidate" if activate else "base",
                    "policy_conditions": policy.get("conditions", []),
                    "feature_values": feature_values,
                    f"{stage}_te_cm": chosen_te,
                    f"{stage}_re_deg": _safe_float(chosen.get(f"{stage}_re_deg")),
                    f"{stage}_inliers": _safe_float(chosen.get(f"{stage}_inliers")),
                    "base_te_cm": base_te,
                    "candidate_te_cm": _safe_float(candidate.get(f"{stage}_te_cm")),
                    "delta_te_cm": delta,
                    "label": label,
                }
            )
    paired_summary = {
        "query_count": len(decisions),
        "activated_count": sum(1 for row in decisions if row["activated"]),
        "helped_count": sum(1 for row in decisions if row["label"] == "helped"),
        "harmed_count": sum(1 for row in decisions if row["label"] == "harmed"),
        "unchanged_count": sum(1 for row in decisions if row["label"] == "unchanged"),
        "regression_20cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) >= 20.0),
        "regression_50cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) >= 50.0),
        "improvement_20cm_count": sum(1 for row in decisions if (row.get("delta_te_cm") or 0.0) <= -20.0),
        "mean_delta_te_cm": _mean(row.get("delta_te_cm") for row in decisions),
    }
    return {
        "summary": summarize_rows(decisions, stage=stage),
        "paired": paired_summary,
        "folds": fold_reports,
        "decisions": decisions,
        "policy": {
            "type": "cross_validated_feature_rule_gate",
            "features": list(features),
            "folds": folds,
            "max_depth": int(max_depth),
            "min_train_queries": min_train_queries,
            "min_train_activated": min_train_activated,
            "min_score": min_score,
            "min_median_gain_cm": min_median_gain_cm,
            "regression_50_penalty": regression_50_penalty,
            "stump_max_train_regression_20": stump_max_train_regression_20,
            "stump_max_train_regression_50": stump_max_train_regression_50,
            "uses_gt_for_fold_labels": True,
            "uses_gt_at_inference": False,
        },
    }


def _macro_summary(scene_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "median_te_cm",
        "median_re_deg",
        "p90_te_cm",
        "p95_te_cm",
        "cvar10_te_cm",
        "severe_rate_1m",
        "catastrophic_rate_5m",
        "avg_inliers",
        *RECALL_THRESHOLDS.keys(),
    )
    out = {"scene_count": len(scene_summaries)}
    for key in keys:
        out[key] = _mean(summary.get(key) for summary in scene_summaries)
    return out


def _macro_paired(paired_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = {
        "query_count": 0,
        "helped_count": 0,
        "harmed_count": 0,
        "unchanged_count": 0,
        "raw_better_count": 0,
        "raw_worse_count": 0,
        "oracle_selected_candidate_count": 0,
        "regression_20cm_count": 0,
        "regression_50cm_count": 0,
        "improvement_20cm_count": 0,
    }
    for summary in paired_summaries:
        for key in total:
            total[key] += int(summary.get(key, 0))
    total["mean_delta_te_cm"] = _mean(summary.get("mean_delta_te_cm") for summary in paired_summaries)
    return total


def _macro_gate_paired(paired_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = {
        "query_count": 0,
        "activated_count": 0,
        "helped_count": 0,
        "harmed_count": 0,
        "unchanged_count": 0,
        "regression_20cm_count": 0,
        "regression_50cm_count": 0,
        "improvement_20cm_count": 0,
    }
    for summary in paired_summaries:
        for key in total:
            total[key] += int(summary.get(key, 0))
    total["mean_delta_te_cm"] = _mean(summary.get("mean_delta_te_cm") for summary in paired_summaries)
    return total


def build_sparse_gate_report(
    *,
    root: Path,
    scenes: Sequence[str],
    base_name: str,
    candidates: Mapping[str, str],
    stage: str = "sparse",
    improvement_margin_cm: float = 1.0,
    learned_gate_folds: int = 5,
    gate_min_train_queries: int = 30,
    stump_min_train_activated: int = 20,
    stump_min_score: float = 0.0,
    stump_min_median_gain_cm: float = 0.0,
    stump_regression_50_penalty: float = 4.0,
    stump_max_train_regression_20: int | None = None,
    stump_max_train_regression_50: int | None = 0,
    feature_rule_depth: int = 2,
) -> dict[str, Any]:
    scene_reports: dict[str, Any] = {}
    all_cases: list[dict[str, Any]] = []
    for scene in scenes:
        base_dir = root / scene / base_name
        if not (base_dir / "results.json").exists():
            scene_reports[scene] = {"status": "missing_base", "base_dir": str(base_dir), "candidates": {}}
            continue
        base_rows = load_sparse_rows(base_dir, stage=stage)
        scene_report: dict[str, Any] = {
            "status": "ok",
            "base_dir": str(base_dir),
            "base": summarize_rows(base_rows, stage=stage),
            "candidates": {},
        }
        for candidate_name, candidate_run_name in candidates.items():
            candidate_dir = root / scene / candidate_run_name
            if not (candidate_dir / "results.json").exists():
                scene_report["candidates"][candidate_name] = {
                    "status": "missing",
                    "candidate_dir": str(candidate_dir),
                }
                continue
            candidate_rows = load_sparse_rows(candidate_dir, stage=stage)
            paired, bins, oracle_rows, case_rows = _paired_report(
                scene=scene,
                candidate_name=candidate_name,
                base_rows=base_rows,
                candidate_rows=candidate_rows,
                stage=stage,
                improvement_margin_cm=improvement_margin_cm,
            )
            learned_gate = build_inlier_bin_gate_cv(
                base_rows,
                candidate_rows,
                stage=stage,
                folds=learned_gate_folds,
                improvement_margin_cm=improvement_margin_cm,
                min_train_queries=gate_min_train_queries,
            )
            feature_stump_gate = build_feature_stump_gate_cv(
                base_rows,
                candidate_rows,
                stage=stage,
                folds=learned_gate_folds,
                improvement_margin_cm=improvement_margin_cm,
                min_train_queries=gate_min_train_queries,
                min_train_activated=stump_min_train_activated,
                min_score=stump_min_score,
                min_median_gain_cm=stump_min_median_gain_cm,
                regression_50_penalty=stump_regression_50_penalty,
                stump_max_train_regression_20=stump_max_train_regression_20,
                stump_max_train_regression_50=stump_max_train_regression_50,
            )
            feature_rule_gate = build_feature_rule_gate_cv(
                base_rows,
                candidate_rows,
                stage=stage,
                folds=learned_gate_folds,
                max_depth=feature_rule_depth,
                improvement_margin_cm=improvement_margin_cm,
                min_train_queries=gate_min_train_queries,
                min_train_activated=stump_min_train_activated,
                min_score=stump_min_score,
                min_median_gain_cm=stump_min_median_gain_cm,
                regression_50_penalty=stump_regression_50_penalty,
                stump_max_train_regression_20=stump_max_train_regression_20,
                stump_max_train_regression_50=stump_max_train_regression_50,
            )
            all_cases.extend(case_rows)
            scene_report["candidates"][candidate_name] = {
                "status": "ok",
                "candidate_dir": str(candidate_dir),
                "candidate": summarize_rows(candidate_rows, stage=stage),
                "oracle": summarize_rows(oracle_rows, stage=stage),
                "learned_gate_cv": learned_gate,
                "feature_stump_gate_cv": feature_stump_gate,
                "feature_rule_gate_cv": feature_rule_gate,
                "paired": paired,
                "confidence_bins": bins,
            }
        scene_reports[scene] = scene_report

    macro: dict[str, Any] = {}
    for candidate_name in candidates:
        base_summaries = []
        candidate_summaries = []
        oracle_summaries = []
        learned_summaries = []
        feature_stump_summaries = []
        feature_rule_summaries = []
        paired_summaries = []
        learned_paired_summaries = []
        feature_stump_paired_summaries = []
        feature_rule_paired_summaries = []
        scenes_available = []
        for scene, scene_report in scene_reports.items():
            candidate_report = scene_report.get("candidates", {}).get(candidate_name, {})
            if candidate_report.get("status") != "ok":
                continue
            scenes_available.append(scene)
            base_summaries.append(scene_report["base"])
            candidate_summaries.append(candidate_report["candidate"])
            oracle_summaries.append(candidate_report["oracle"])
            learned_summaries.append(candidate_report["learned_gate_cv"]["summary"])
            feature_stump_summaries.append(candidate_report["feature_stump_gate_cv"]["summary"])
            feature_rule_summaries.append(candidate_report["feature_rule_gate_cv"]["summary"])
            paired_summaries.append(candidate_report["paired"])
            learned_paired_summaries.append(candidate_report["learned_gate_cv"]["paired"])
            feature_stump_paired_summaries.append(candidate_report["feature_stump_gate_cv"]["paired"])
            feature_rule_paired_summaries.append(candidate_report["feature_rule_gate_cv"]["paired"])
        macro[candidate_name] = {
            "scene_count": len(scenes_available),
            "scenes": scenes_available,
            "base": _macro_summary(base_summaries),
            "candidate": _macro_summary(candidate_summaries),
            "oracle": _macro_summary(oracle_summaries),
            "learned_gate_cv": _macro_summary(learned_summaries),
            "feature_stump_gate_cv": _macro_summary(feature_stump_summaries),
            "feature_rule_gate_cv": _macro_summary(feature_rule_summaries),
            "paired": _macro_paired(paired_summaries),
            "learned_gate_paired": _macro_gate_paired(learned_paired_summaries),
            "feature_stump_gate_paired": _macro_gate_paired(feature_stump_paired_summaries),
            "feature_rule_gate_paired": _macro_gate_paired(feature_rule_paired_summaries),
        }

    return {
        "root": str(root),
        "base_name": base_name,
        "candidates": dict(candidates),
        "stage": stage,
        "improvement_margin_cm": float(improvement_margin_cm),
        "feature_stump_gate_params": {
            "min_train_activated": int(stump_min_train_activated),
            "min_score": float(stump_min_score),
            "min_median_gain_cm": float(stump_min_median_gain_cm),
            "regression_50_penalty": float(stump_regression_50_penalty),
            "stump_max_train_regression_20": stump_max_train_regression_20,
            "stump_max_train_regression_50": stump_max_train_regression_50,
            "feature_rule_depth": int(feature_rule_depth),
        },
        "scene_reports": scene_reports,
        "macro": macro,
        "oracle_cases": all_cases,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "scene",
        "candidate",
        "query_id",
        "image_name",
        "base_te_cm",
        "candidate_te_cm",
        "delta_te_cm",
        "base_re_deg",
        "candidate_re_deg",
        "base_inliers",
        "candidate_inliers",
        "confidence_bin",
        "label",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any, digits: int = 3, pct: bool = False) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    if pct:
        number *= 100.0
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Sparse Oracle Gate Report",
        "",
        f"- stage: `{report['stage']}`",
        f"- root: `{report['root']}`",
        f"- base: `{report['base_name']}`",
        f"- improvement margin: `{report['improvement_margin_cm']}cm`",
        "",
        "## Macro",
    ]
    macro_rows: list[list[str]] = []
    for name, row in report["macro"].items():
        base = row["base"]
        cand = row["candidate"]
        oracle = row["oracle"]
        paired = row["paired"]
        macro_rows.append(
            [
                name,
                str(row["scene_count"]),
                _fmt(base.get("median_te_cm")),
                _fmt(cand.get("median_te_cm")),
                _fmt(oracle.get("median_te_cm")),
                _fmt(row["learned_gate_cv"].get("median_te_cm")),
                _fmt(row["feature_stump_gate_cv"].get("median_te_cm")),
                _fmt(base.get("R10cm5deg"), pct=True),
                _fmt(cand.get("R10cm5deg"), pct=True),
                _fmt(oracle.get("R10cm5deg"), pct=True),
                _fmt(row["learned_gate_cv"].get("R10cm5deg"), pct=True),
                _fmt(row["feature_stump_gate_cv"].get("R10cm5deg"), pct=True),
                str(paired.get("helped_count", 0)),
                str(paired.get("harmed_count", 0)),
                str(row["learned_gate_paired"].get("activated_count", 0)),
                str(row["feature_stump_gate_paired"].get("activated_count", 0)),
                str(paired.get("regression_20cm_count", 0)),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "candidate",
                "scenes",
                "base med",
                "cand med",
                "oracle med",
                "bin gate med",
                "stump gate med",
                "base R10%",
                "cand R10%",
                "oracle R10%",
                "bin gate R10%",
                "stump gate R10%",
                "helped",
                "harmed",
                "bin active",
                "stump active",
                "reg20",
            ],
            macro_rows,
        )
    )
    lines.extend(["", "## Per Scene"])
    for scene, scene_report in report["scene_reports"].items():
        lines.extend(["", f"### {scene}"])
        if scene_report.get("status") != "ok":
            lines.append(f"- status: `{scene_report.get('status')}`")
            continue
        rows: list[list[str]] = []
        for name, candidate_report in scene_report.get("candidates", {}).items():
            if candidate_report.get("status") != "ok":
                rows.append([name, candidate_report.get("status", "missing"), "-", "-", "-", "-", "-", "-"])
                continue
            paired = candidate_report["paired"]
            rows.append(
                [
                    name,
                    "ok",
                    _fmt(scene_report["base"].get("median_te_cm")),
                    _fmt(candidate_report["candidate"].get("median_te_cm")),
                    _fmt(candidate_report["oracle"].get("median_te_cm")),
                    _fmt(candidate_report["learned_gate_cv"]["summary"].get("median_te_cm")),
                    _fmt(candidate_report["feature_stump_gate_cv"]["summary"].get("median_te_cm")),
                    _fmt(candidate_report["oracle"].get("R5cm5deg"), pct=True),
                    _fmt(candidate_report["learned_gate_cv"]["summary"].get("R5cm5deg"), pct=True),
                    _fmt(candidate_report["feature_stump_gate_cv"]["summary"].get("R5cm5deg"), pct=True),
                    str(paired.get("helped_count", 0)),
                    str(paired.get("harmed_count", 0)),
                    str(candidate_report["learned_gate_cv"]["paired"].get("activated_count", 0)),
                    str(candidate_report["feature_stump_gate_cv"]["paired"].get("activated_count", 0)),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "candidate",
                    "status",
                    "base med",
                    "cand med",
                    "oracle med",
                    "bin gate med",
                    "stump gate med",
                    "oracle R5%",
                    "bin gate R5%",
                    "stump gate R5%",
                    "helped",
                    "harmed",
                    "bin active",
                    "stump active",
                ],
                rows,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_text(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"git unavailable: {exc}\n"
    return result.stdout


def _write_artifacts(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    argv: Sequence[str],
    source_split: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "report.md", report)
    _write_csv(output_dir / "oracle_cases.csv", report.get("oracle_cases", []))
    command = "python -m loc_gs.scripts.build_sparse_gate_report " + " ".join(shlex.quote(arg) for arg in argv)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    git_commit = _git_text(["rev-parse", "HEAD"]).strip()
    manifest = {
        "git_commit": git_commit,
        "command": command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_split": source_split,
        "stage": report.get("stage"),
        "root": report.get("root"),
        "base_name": report.get("base_name"),
        "candidates": report.get("candidates"),
        "scene_count": len(report.get("scene_reports", {})),
        "uses_gt_for_method_selection": False,
        "oracle_upper_bound_only": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_audit = {
        "audit_status": "oracle_gate_source_split_recorded",
        "source_split": source_split,
        "paper_safe_for_tuning": str(source_split).lower() not in {"test", "official_test"},
        "uses_test_for_tuning": str(source_split).lower() in {"test", "official_test"},
    }
    (output_dir / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_text(["status", "--short"]), encoding="utf-8")


def _parse_candidate(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"candidate must be NAME=RUN_NAME, got {spec!r}")
    name, run_name = spec.split("=", 1)
    name = name.strip()
    run_name = run_name.strip()
    if not name or not run_name:
        raise ValueError(f"candidate must be NAME=RUN_NAME, got {spec!r}")
    return name, run_name


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-only sparse oracle-gate upper-bound diagnostics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=RUN_NAME under ROOT/SCENE/")
    parser.add_argument("--stage", default="sparse", choices=("sparse", "dense"))
    parser.add_argument("--improvement-margin-cm", type=float, default=1.0)
    parser.add_argument("--learned-gate-folds", type=int, default=5)
    parser.add_argument("--gate-min-train-queries", type=int, default=30)
    parser.add_argument("--stump-min-train-activated", type=int, default=20)
    parser.add_argument("--stump-min-score", type=float, default=0.0)
    parser.add_argument("--stump-min-median-gain-cm", type=float, default=0.0)
    parser.add_argument("--stump-regression-50-penalty", type=float, default=4.0)
    parser.add_argument(
        "--stump-max-train-regression-20",
        type=int,
        default=-1,
        help="Reject feature stumps with more than this many >=20cm train regressions. Use -1 to disable.",
    )
    parser.add_argument(
        "--stump-max-train-regression-50",
        type=int,
        default=0,
        help="Reject feature stumps with more than this many >=50cm train regressions. Use -1 to disable.",
    )
    parser.add_argument("--feature-rule-depth", type=int, default=2)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_argparser().parse_args(raw_argv)
    candidates = dict(_parse_candidate(spec) for spec in args.candidate)
    scenes = tuple(args.scene) if args.scene else tuple(scene for scene in CAMBRIDGE_SCENES if (args.root / scene).exists())
    report = build_sparse_gate_report(
        root=args.root,
        scenes=scenes,
        base_name=args.base_name,
        candidates=candidates,
        stage=args.stage,
        improvement_margin_cm=args.improvement_margin_cm,
        learned_gate_folds=args.learned_gate_folds,
        gate_min_train_queries=args.gate_min_train_queries,
        stump_min_train_activated=args.stump_min_train_activated,
        stump_min_score=args.stump_min_score,
        stump_min_median_gain_cm=args.stump_min_median_gain_cm,
        stump_regression_50_penalty=args.stump_regression_50_penalty,
        stump_max_train_regression_20=None
        if args.stump_max_train_regression_20 < 0
        else args.stump_max_train_regression_20,
        stump_max_train_regression_50=None
        if args.stump_max_train_regression_50 < 0
        else args.stump_max_train_regression_50,
        feature_rule_depth=args.feature_rule_depth,
    )
    _write_artifacts(output_dir=args.output_dir, report=report, argv=raw_argv, source_split=args.source_split)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "stage": args.stage,
                "macro": {
                    name: {
                        "scene_count": row["scene_count"],
                        "base_median_te_cm": row["base"].get("median_te_cm"),
                        "candidate_median_te_cm": row["candidate"].get("median_te_cm"),
                        "oracle_median_te_cm": row["oracle"].get("median_te_cm"),
                        "learned_gate_median_te_cm": row["learned_gate_cv"].get("median_te_cm"),
                        "feature_stump_gate_median_te_cm": row["feature_stump_gate_cv"].get("median_te_cm"),
                        "feature_rule_gate_median_te_cm": row["feature_rule_gate_cv"].get("median_te_cm"),
                        "helped_count": row["paired"].get("helped_count"),
                        "harmed_count": row["paired"].get("harmed_count"),
                        "learned_gate_activated_count": row["learned_gate_paired"].get("activated_count"),
                        "feature_stump_gate_activated_count": row["feature_stump_gate_paired"].get("activated_count"),
                        "feature_rule_gate_activated_count": row["feature_rule_gate_paired"].get("activated_count"),
                    }
                    for name, row in report["macro"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
