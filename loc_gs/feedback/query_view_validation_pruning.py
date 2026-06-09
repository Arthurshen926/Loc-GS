from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "query_view_validation_pruning_v1"


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) else None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _query_id(row: Mapping[str, Any]) -> str:
    return str(row.get("query_id", row.get("image_name", row.get("image_id", "")))).strip()


def _translation_cm(row: Mapping[str, Any]) -> float | None:
    for key in ("sparse_te_cm", "sparse_TE", "translation_cm", "te_cm"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    sparse = row.get("sparse")
    if isinstance(sparse, Mapping):
        for key in ("TE", "te", "te_cm"):
            value = _safe_float(sparse.get(key))
            if value is not None:
                return value
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load sparse eval rows from Loc-GS/ULF result artifacts."""

    root = Path(run_dir)
    payload = _load_json(root / "results.json")
    if isinstance(payload, Mapping):
        if isinstance(payload.get("rows"), list):
            payload = payload["rows"]
        elif isinstance(payload.get("results"), list):
            payload = payload["results"]
    if not isinstance(payload, list):
        raise ValueError(f"eval results must be a list or contain rows/results: {root / 'results.json'}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def _reject_test_split(payload: Mapping[str, Any]) -> None:
    for key in ("split_name", "split", "source_split_name", "feedback_bank_split_name"):
        value = str(payload.get(key, "")).strip().lower()
        if value == "test" or value.endswith("_test"):
            raise ValueError("test split is not allowed for query/view validation pruning")


def _active_plan_lookup(active_plan: Mapping[str, Any]) -> dict[int, list[str]]:
    raw = active_plan.get("landmark_fusion_plan", active_plan.get("descriptor_fusion", {}).get("landmark_fusion_plan", {}))
    if not isinstance(raw, Mapping):
        raise ValueError("active plan must contain landmark_fusion_plan")
    out: dict[int, list[str]] = {}
    for raw_gid, raw_row in raw.items():
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            continue
        if gid < 0 or not isinstance(raw_row, Mapping):
            continue
        views = raw_row.get("selected_view_ids", raw_row.get("view_ids", []))
        if isinstance(views, str):
            selected = [views] if views.strip() else []
        elif isinstance(views, Iterable):
            selected = [str(view) for view in views if str(view).strip()]
        else:
            selected = []
        deduped = sorted(set(selected))
        if deduped:
            out[gid] = deduped
    return out


def _match_records(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("sparse_match_attributions", "match_attributions", "matches"):
        raw = row.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, Mapping)]
    return []


def _match_gaussian_id(record: Mapping[str, Any]) -> int | None:
    for key in ("matched_gaussian_id", "gaussian_id", "landmark_id"):
        try:
            gid = int(record.get(key))
        except (TypeError, ValueError):
            continue
        if gid >= 0:
            return gid
    return None


def _match_confidence(record: Mapping[str, Any]) -> float:
    descriptor = _safe_float(record.get("descriptor_score"))
    detector = _safe_float(record.get("detector_score"))
    reproj = _safe_float(record.get("reprojection_error_px"))
    descriptor_factor = 1.0 if descriptor is None else max(0.05, min(1.0, abs(descriptor)))
    detector_factor = 1.0 if detector is None else max(0.05, min(1.0, detector))
    if reproj is None:
        reproj_factor = 1.0
    else:
        reproj_factor = max(0.1, min(1.0, math.exp(-max(0.0, reproj) / 8.0)))
    return float(descriptor_factor * detector_factor * reproj_factor)


def _risk_confidence(record: Mapping[str, Any]) -> float:
    base = _match_confidence(record)
    reproj = _safe_float(record.get("reprojection_error_px"))
    is_inlier = _safe_bool(record.get("pnp_inlier", record.get("inlier", False)))
    if is_inlier and (reproj is None or reproj <= 4.0):
        return 0.0
    if reproj is None:
        reproj_factor = 0.5
    else:
        reproj_factor = max(0.25, min(2.0, max(0.0, reproj) / 8.0))
    inlier_factor = 0.5 if is_inlier else 1.0
    return float(base * reproj_factor * inlier_factor)


def _benefit_confidence(record: Mapping[str, Any]) -> float:
    if not _safe_bool(record.get("pnp_inlier", record.get("inlier", False))):
        return 0.0
    return _match_confidence(record)


def _active_plan_from_lookup(
    lookup: Mapping[int, list[str]],
    *,
    source_plan: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "split_name": str(source_plan.get("split_name", source_plan.get("split", "unknown"))),
        "landmark_fusion_plan": {
            str(gid): {"selected_view_ids": list(views)}
            for gid, views in sorted(lookup.items())
            if views
        },
        "metadata": dict(metadata),
    }


def prune_active_fusion_plan_from_sparse_validation(
    *,
    active_plan: Mapping[str, Any],
    baseline_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    protected_te_cm: float = 15.0,
    hard_te_cm: float = 20.0,
    regression_margin_cm: float = 20.0,
    improvement_margin_cm: float = 20.0,
    min_pair_score: float = 0.0,
    min_views_per_landmark: int = 0,
) -> dict[str, Any]:
    """Prune active descriptor-fusion pairs using real sparse validation deltas.

    This is an offline artifact transform. It never changes per-query inference:
    it only removes active `(landmark, view)` descriptor-fusion sources that are
    repeatedly attributed to sparse regressions and not to sparse improvements.
    """

    _reject_test_split(active_plan)
    lookup = _active_plan_lookup(active_plan)
    baseline_by_query = {_query_id(row): dict(row) for row in baseline_rows if _query_id(row)}
    candidate_by_query = {_query_id(row): dict(row) for row in candidate_rows if _query_id(row)}

    pair_benefit: dict[tuple[int, str], float] = {}
    pair_risk: dict[tuple[int, str], float] = {}
    regression_query_count = 0
    improvement_query_count = 0
    attributed_match_count = 0

    for query_id, candidate in candidate_by_query.items():
        baseline = baseline_by_query.get(query_id)
        if baseline is None:
            continue
        base_te = _translation_cm(baseline)
        cand_te = _translation_cm(candidate)
        if base_te is None or cand_te is None:
            continue
        delta = float(cand_te - base_te)
        is_protected_regression = base_te <= float(protected_te_cm) and delta >= float(regression_margin_cm)
        is_hard_improvement = base_te >= float(hard_te_cm) and -delta >= float(improvement_margin_cm)
        if is_protected_regression:
            regression_query_count += 1
        if is_hard_improvement:
            improvement_query_count += 1
        if not (is_protected_regression or is_hard_improvement):
            continue
        for record in _match_records(candidate):
            gid = _match_gaussian_id(record)
            if gid is None or gid not in lookup:
                continue
            attributed_match_count += 1
            for view_id in lookup[gid]:
                key = (int(gid), str(view_id))
                if is_protected_regression:
                    pair_risk[key] = float(pair_risk.get(key, 0.0) + _risk_confidence(record))
                if is_hard_improvement:
                    pair_benefit[key] = float(pair_benefit.get(key, 0.0) + _benefit_confidence(record))

    attribution_status = "attributed" if attributed_match_count > 0 else "metric_only"
    pruned_lookup: dict[int, list[str]] = {}
    removed_pairs: list[dict[str, Any]] = []
    protected_pair_count = 0
    for gid, views in sorted(lookup.items()):
        kept: list[str] = []
        removable: list[tuple[str, float, float, float]] = []
        for view_id in views:
            key = (int(gid), str(view_id))
            benefit = float(pair_benefit.get(key, 0.0))
            risk = float(pair_risk.get(key, 0.0))
            score = benefit - risk
            if attribution_status == "attributed" and risk > 0.0 and score < float(min_pair_score):
                removable.append((str(view_id), benefit, risk, score))
            else:
                kept.append(str(view_id))
        min_keep = max(0, int(min_views_per_landmark))
        if min_keep > 0 and len(kept) < min_keep:
            for view_id, benefit, risk, score in sorted(removable, key=lambda item: item[3], reverse=True):
                if len(kept) >= min_keep:
                    break
                kept.append(view_id)
                protected_pair_count += 1
                removable = [item for item in removable if item[0] != view_id]
        for view_id, benefit, risk, score in removable:
            removed_pairs.append(
                {
                    "gaussian_id": int(gid),
                    "view_id": str(view_id),
                    "benefit": float(benefit),
                    "risk": float(risk),
                    "score": float(score),
                }
            )
        if kept:
            pruned_lookup[int(gid)] = sorted(set(kept))

    original_pair_count = int(sum(len(views) for views in lookup.values()))
    kept_pair_count = int(sum(len(views) for views in pruned_lookup.values()))
    metadata = {
        "source_schema_version": str(active_plan.get("schema_version", "")),
        "original_landmark_count": int(len(lookup)),
        "pruned_landmark_count": int(len(pruned_lookup)),
        "original_pair_count": original_pair_count,
        "kept_pair_count": kept_pair_count,
        "removed_pair_count": int(len(removed_pairs)),
        "protected_pair_count": int(protected_pair_count),
        "regression_query_count": int(regression_query_count),
        "improvement_query_count": int(improvement_query_count),
        "attributed_match_count": int(attributed_match_count),
        "attribution_status": attribution_status,
        "protected_te_cm": float(protected_te_cm),
        "hard_te_cm": float(hard_te_cm),
        "regression_margin_cm": float(regression_margin_cm),
        "improvement_margin_cm": float(improvement_margin_cm),
        "min_pair_score": float(min_pair_score),
        "min_views_per_landmark": int(min_views_per_landmark),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "active_fusion_plan": _active_plan_from_lookup(pruned_lookup, source_plan=active_plan, metadata=metadata),
        "removed_pairs": removed_pairs,
        "metrics": metadata,
    }
