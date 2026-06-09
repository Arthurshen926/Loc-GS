from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from loc_gs.feedback.audit import audit_feedback_bank_v2, audit_feedback_bank_v3
from loc_gs.feedback.io import load_feedback_bank


_LOCALIZE_RE = re.compile(r"^\s*Localize image:\s*(?P<image>.+?)\s*$")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    w = pos - lo
    return float((1.0 - w) * ordered[lo] + w * ordered[hi])


def _parse_log_image_names(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    names: list[str] = []
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOCALIZE_RE.match(raw)
        if match:
            names.append(match.group("image").strip())
    return names


def _stage_metric(row: Mapping[str, Any], stage: str, key: str) -> float | None:
    aliases = {
        "te": (f"{stage}_TE", f"{stage}_te", f"{stage}_te_cm"),
        "re": (f"{stage}_AE", f"{stage}_ae", f"{stage}_re_deg"),
    }[key]
    for alias in aliases:
        value = _safe_float(row.get(alias))
        if value is not None:
            return value
    nested = row.get(stage)
    if isinstance(nested, Mapping):
        nested_aliases = {"te": ("TE", "te", "te_cm"), "re": ("AE", "ae", "re_deg")}[key]
        for alias in nested_aliases:
            value = _safe_float(nested.get(alias))
            if value is not None:
                return value
    return None


def _stage_inliers(row: Mapping[str, Any], stage: str) -> int | None:
    for key in (f"{stage}_inliers", "inliers"):
        value = _safe_int(row.get(key))
        if value is not None:
            return value
    nested = row.get(stage)
    if isinstance(nested, Mapping):
        return _safe_int(nested.get("inliers"))
    return None


def load_ulfloc_sparse_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load ULF-Loc sparse rows with stable query ids.

    Upstream ULF-Loc ``results.json`` often omits image names.  The evaluator
    writes them to ``output.log`` in the same order, so this loader joins the two
    files and falls back to deterministic query indices only when the log is
    unavailable.
    """

    root = Path(run_dir)
    results_path = root / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"missing ULF-Loc results.json: {results_path}")
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        if isinstance(raw.get("results"), list):
            raw = raw["results"]
        elif isinstance(raw.get("rows"), list):
            raw = raw["rows"]
    if not isinstance(raw, list):
        raise ValueError(f"ULF-Loc results must be a list, {{results: list}}, or {{rows: list}}: {results_path}")
    log_names = _parse_log_image_names(root / "output.log")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        image_name = item.get("image_name")
        if not image_name and index < len(log_names):
            image_name = log_names[index]
        query_id = str(image_name or f"query_{index:06d}")
        te = _stage_metric(item, "sparse", "te")
        re_deg = _stage_metric(item, "sparse", "re")
        rows.append(
            {
                "query_id": query_id,
                "query_index": int(index),
                "image_name": query_id,
                "sparse_te_cm": te,
                "sparse_re_deg": re_deg,
                "sparse_inliers": _stage_inliers(item, "sparse"),
                "localized": bool(te is not None and re_deg is not None),
            }
        )
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tes = [float(row["sparse_te_cm"]) for row in rows if row.get("sparse_te_cm") is not None]
    res = [float(row["sparse_re_deg"]) for row in rows if row.get("sparse_re_deg") is not None]
    return {
        "query_count": int(len(rows)),
        "valid_count": int(len(tes)),
        "median_te_cm": _median(tes),
        "median_re_deg": _median(res),
        "p90_te_cm": _percentile(tes, 0.90),
        "p95_te_cm": _percentile(tes, 0.95),
        "severe_rate_1m": float(sum(1 for value in tes if value > 100.0) / len(tes)) if tes else None,
        "catastrophic_rate_5m": float(sum(1 for value in tes if value > 500.0) / len(tes)) if tes else None,
    }


def _feedback_split_name(bank: Mapping[str, Any]) -> str:
    manifest = bank.get("manifest", {})
    if not isinstance(manifest, Mapping):
        return "unknown"
    split = str(manifest.get("split_name", manifest.get("split", "unknown"))).strip()
    return split or "unknown"


def _feedback_schema_version(bank: Mapping[str, Any]) -> str:
    manifest = bank.get("manifest", {})
    if not isinstance(manifest, Mapping):
        return "unknown"
    return str(manifest.get("schema_version", "")).strip()


def _audit_feedback_bank_for_sparse_attribution(path: str | Path) -> dict[str, Any]:
    bank = load_feedback_bank(path)
    schema = _feedback_schema_version(bank)
    if schema == "feedback_bank_v3":
        return audit_feedback_bank_v3(path)
    return audit_feedback_bank_v2(path)


def _record_gaussian_id(record: Mapping[str, Any]) -> int | None:
    raw = record.get("matched_gaussian_id", record.get("gaussian_id"))
    try:
        gid = int(raw)
    except (TypeError, ValueError):
        return None
    return gid if gid >= 0 else None


def _record_query_id(record: Mapping[str, Any]) -> str:
    return str(record.get("query_id", record.get("image_id", ""))).strip()


def _record_reliability(record: Mapping[str, Any]) -> float:
    reliability = 1.0
    reproj = _safe_float(record.get("reprojection_error_px"))
    if reproj is not None:
        reliability *= max(0.0, min(1.0, 1.0 - reproj / 16.0))
    descriptor = _safe_float(record.get("descriptor_score"))
    if descriptor is not None:
        reliability *= max(0.1, min(1.0, abs(descriptor)))
    return float(max(0.0, min(1.0, reliability)))


def _record_negative_reliability(record: Mapping[str, Any]) -> float:
    """Score high-confidence wrong correspondences for regression attribution."""

    descriptor = _safe_float(record.get("descriptor_score"))
    descriptor_factor = 1.0 if descriptor is None else max(0.1, min(1.0, abs(float(descriptor))))
    margin = _safe_float(record.get("descriptor_margin"))
    margin_factor = 1.0 if margin is None else max(0.1, min(1.0, 1.0 / (1.0 + max(0.0, float(margin)))))
    detector = _safe_float(record.get("detector_score"))
    detector_factor = 1.0 if detector is None else max(0.1, min(1.0, float(detector)))
    reproj = _safe_float(record.get("reprojection_error_px"))
    reproj_factor = 0.5 if reproj is None else max(0.25, min(1.0, float(reproj) / 16.0))
    local_geometry = _safe_float(record.get("local_geometry_score"))
    geometry_factor = 1.0 if local_geometry is None else max(0.1, min(1.0, float(local_geometry)))
    return float(max(0.0, min(1.0, descriptor_factor * margin_factor * detector_factor * reproj_factor * geometry_factor)))


def _vector_observation(record: Mapping[str, Any], key: str, *, length: int) -> list[float] | None:
    raw = record.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) < int(length):
        return None
    values: list[float] = []
    for idx in range(int(length)):
        value = _safe_float(raw[idx])
        if value is None:
            return None
        values.append(float(value))
    return values


def _record_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    xy_norm = _vector_observation(record, "query_xy_norm", length=2)
    if xy_norm is not None:
        out["xy_norm"] = xy_norm
    bearing = _vector_observation(record, "bearing", length=3)
    if bearing is not None:
        out["bearing"] = bearing
    camera_xyz = _vector_observation(record, "camera_xyz", length=3)
    if camera_xyz is not None:
        out["camera_xyz"] = camera_xyz
    for source_key, output_key in (
        ("depth_m", "depth_m"),
        ("reprojection_error_px", "reprojection_error_px"),
        ("descriptor_margin", "descriptor_margin"),
        ("local_geometry_score", "local_geometry_score"),
        ("ray_artifact_score", "ray_artifact_score"),
        ("ray_entropy", "ray_entropy"),
    ):
        value = _safe_float(record.get(source_key))
        if value is not None:
            out[output_key] = float(value)
    raw_cell = record.get("image_cell")
    if isinstance(raw_cell, (list, tuple)) and len(raw_cell) >= 2:
        cell_x = _safe_int(raw_cell[0])
        cell_y = _safe_int(raw_cell[1])
        if cell_x is not None and cell_y is not None:
            out["image_cell"] = [int(cell_x), int(cell_y)]
    depth_bin = _safe_int(record.get("depth_bin"))
    if depth_bin is not None:
        out["depth_bin"] = int(depth_bin)
    return out


def _landmark_regression_risk_from_feedback_bank(
    feedback_bank: str | Path,
    *,
    regressed_query_delta: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    bank_path = Path(feedback_bank)
    audit = _audit_feedback_bank_for_sparse_attribution(bank_path)
    if audit["audit_status"] != "passed":
        raise ValueError(f"candidate feedback bank audit failed: {audit['reasons']}")
    bank = load_feedback_bank(bank_path)
    split_name = _feedback_split_name(bank)
    if split_name.lower() == "test":
        raise ValueError("test split feedback bank is not allowed for sparse PnP validation")
    regressed = {str(query): float(delta) for query, delta in regressed_query_delta.items() if float(delta) > 0.0}
    risk: dict[str, float] = {}
    per_query_risk: dict[str, dict[str, float]] = {}
    used_records = 0
    for raw_record in bank.get("records", []):
        if not isinstance(raw_record, Mapping):
            continue
        query_id = _record_query_id(raw_record)
        delta = regressed.get(query_id)
        if delta is None:
            continue
        gid = _record_gaussian_id(raw_record)
        if gid is None:
            continue
        used_records += 1
        if bool(raw_record.get("pnp_inlier", False)):
            reliability = _record_reliability(raw_record)
        else:
            reliability = 0.5 * _record_negative_reliability(raw_record)
        value = float(delta * reliability)
        if value <= 0.0:
            continue
        risk[str(gid)] = float(risk.get(str(gid), 0.0) + value)
        query_risk = per_query_risk.setdefault(query_id, {})
        query_risk[str(gid)] = float(query_risk.get(str(gid), 0.0) + value)
    metadata = {
        "candidate_feedback_bank": str(bank_path),
        "candidate_feedback_bank_split_name": split_name,
        "candidate_feedback_bank_record_count": int(len(bank.get("records", []))),
        "candidate_feedback_bank_used_regression_record_count": int(used_records),
        "candidate_feedback_bank_audit": audit,
    }
    return risk, per_query_risk, metadata


def _protected_support_from_feedback_bank(
    feedback_bank: str | Path,
    *,
    protected_query_delta: Mapping[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    bank_path = Path(feedback_bank)
    audit = _audit_feedback_bank_for_sparse_attribution(bank_path)
    if audit["audit_status"] != "passed":
        raise ValueError(f"baseline feedback bank audit failed: {audit['reasons']}")
    bank = load_feedback_bank(bank_path)
    split_name = _feedback_split_name(bank)
    if split_name.lower() == "test":
        raise ValueError("test split feedback bank is not allowed for sparse PnP validation")
    protected = {str(query): float(delta) for query, delta in protected_query_delta.items() if float(delta) > 0.0}
    support: dict[str, dict[str, float]] = {}
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    used_records = 0
    for raw_record in bank.get("records", []):
        if not isinstance(raw_record, Mapping):
            continue
        query_id = _record_query_id(raw_record)
        delta = protected.get(query_id)
        if delta is None:
            continue
        if not bool(raw_record.get("pnp_inlier", False)):
            continue
        gid = _record_gaussian_id(raw_record)
        if gid is None:
            continue
        used_records += 1
        value = float(delta * _record_reliability(raw_record))
        if value <= 0.0:
            continue
        query_support = support.setdefault(query_id, {})
        query_support[str(gid)] = float(query_support.get(str(gid), 0.0) + value)
        observation = _record_observation(raw_record)
        if observation:
            observations.setdefault(query_id, {})[str(gid)] = observation
    metadata = {
        "baseline_feedback_bank": str(bank_path),
        "baseline_feedback_bank_split_name": split_name,
        "baseline_feedback_bank_record_count": int(len(bank.get("records", []))),
        "baseline_feedback_bank_used_inlier_record_count": int(used_records),
        "baseline_feedback_bank_audit": audit,
    }
    return support, observations, metadata


def _validated_support_from_feedback_bank(
    feedback_bank: str | Path,
    *,
    validated_query_delta: Mapping[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    bank_path = Path(feedback_bank)
    audit = _audit_feedback_bank_for_sparse_attribution(bank_path)
    if audit["audit_status"] != "passed":
        raise ValueError(f"candidate feedback bank audit failed: {audit['reasons']}")
    bank = load_feedback_bank(bank_path)
    split_name = _feedback_split_name(bank)
    if split_name.lower() == "test":
        raise ValueError("test split feedback bank is not allowed for sparse PnP validation")
    validated = {str(query): float(delta) for query, delta in validated_query_delta.items() if float(delta) > 0.0}
    support: dict[str, dict[str, float]] = {}
    observations: dict[str, dict[str, dict[str, Any]]] = {}
    used_records = 0
    for raw_record in bank.get("records", []):
        if not isinstance(raw_record, Mapping):
            continue
        query_id = _record_query_id(raw_record)
        delta = validated.get(query_id)
        if delta is None:
            continue
        if not bool(raw_record.get("pnp_inlier", False)):
            continue
        gid = _record_gaussian_id(raw_record)
        if gid is None:
            continue
        used_records += 1
        value = float(delta * _record_reliability(raw_record))
        if value <= 0.0:
            continue
        query_support = support.setdefault(query_id, {})
        query_support[str(gid)] = float(query_support.get(str(gid), 0.0) + value)
        observation = _record_observation(raw_record)
        if observation:
            observations.setdefault(query_id, {})[str(gid)] = observation
    metadata = {
        "validated_feedback_bank": str(bank_path),
        "validated_feedback_bank_split_name": split_name,
        "validated_feedback_bank_record_count": int(len(bank.get("records", []))),
        "validated_feedback_bank_used_inlier_record_count": int(used_records),
        "validated_feedback_bank_audit": audit,
    }
    return support, observations, metadata


def build_sparse_pnp_validation_profile(
    *,
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    scene: str,
    split_name: str,
    protected_te_cm: float = 15.0,
    hard_te_cm: float = 20.0,
    regression_margin_cm: float = 20.0,
    improvement_margin_cm: float = 20.0,
    baseline_feedback_bank: str | Path | None = None,
    candidate_feedback_bank: str | Path | None = None,
    protect_all_baseline_good_queries: bool = False,
) -> dict[str, Any]:
    split = str(split_name).strip() or "unknown"
    if split.lower() == "test":
        raise ValueError("test split results are not allowed for sparse PnP validation")
    baseline_rows = load_ulfloc_sparse_rows(baseline_run_dir)
    candidate_rows = load_ulfloc_sparse_rows(candidate_run_dir)
    candidate_by_query = {str(row["query_id"]): row for row in candidate_rows}

    paired: list[dict[str, Any]] = []
    regression_queries: list[str] = []
    protected_regressions: list[str] = []
    hard_improvements: list[str] = []
    regressed_delta: dict[str, float] = {}
    hard_improvement_delta: dict[str, float] = {}
    baseline_good_support_weight: dict[str, float] = {}
    for base in baseline_rows:
        query_id = str(base["query_id"])
        cand = candidate_by_query.get(query_id)
        if cand is None:
            continue
        base_te = _safe_float(base.get("sparse_te_cm"))
        cand_te = _safe_float(cand.get("sparse_te_cm"))
        base_re = _safe_float(base.get("sparse_re_deg"))
        cand_re = _safe_float(cand.get("sparse_re_deg"))
        delta = None if base_te is None or cand_te is None else float(cand_te - base_te)
        label = "unknown" if delta is None else "neutral"
        if delta is not None and delta >= float(regression_margin_cm):
            label = "regression"
            regression_queries.append(query_id)
            regressed_delta[query_id] = float(delta)
            if base_te is not None and base_te <= float(protected_te_cm):
                protected_regressions.append(query_id)
        elif delta is not None and -delta >= float(improvement_margin_cm):
            label = "improvement"
            if base_te is not None and base_te >= float(hard_te_cm):
                hard_improvements.append(query_id)
                hard_improvement_delta[query_id] = float(-delta)
        if (
            bool(protect_all_baseline_good_queries)
            and base_te is not None
            and base_te <= float(protected_te_cm)
        ):
            baseline_good_support_weight[query_id] = max(1.0, float(protected_te_cm) - float(base_te))
        paired.append(
            {
                "query_id": query_id,
                "baseline_sparse_te_cm": base_te,
                "candidate_sparse_te_cm": cand_te,
                "delta_sparse_te_cm": delta,
                "baseline_sparse_re_deg": base_re,
                "candidate_sparse_re_deg": cand_re,
                "baseline_sparse_inliers": base.get("sparse_inliers"),
                "candidate_sparse_inliers": cand.get("sparse_inliers"),
                "label": label,
                "protected_by_baseline": bool(base_te is not None and base_te <= float(protected_te_cm)),
                "hard_by_baseline": bool(base_te is not None and base_te >= float(hard_te_cm)),
            }
        )

    landmark_risk: dict[str, float] = {}
    regression_per_query_risk: dict[str, dict[str, float]] = {}
    feedback_metadata: dict[str, Any] = {"candidate_feedback_bank": None}
    if candidate_feedback_bank is not None:
        landmark_risk, regression_per_query_risk, feedback_metadata = _landmark_regression_risk_from_feedback_bank(
            candidate_feedback_bank,
            regressed_query_delta=regressed_delta,
        )
    validated_support: dict[str, dict[str, float]] = {}
    validated_observations: dict[str, dict[str, dict[str, Any]]] = {}
    validated_feedback_metadata: dict[str, Any] = {"validated_feedback_bank": None}
    if candidate_feedback_bank is not None:
        validated_support, validated_observations, validated_feedback_metadata = _validated_support_from_feedback_bank(
            candidate_feedback_bank,
            validated_query_delta=hard_improvement_delta,
        )
    protected_delta = dict(baseline_good_support_weight)
    for query_id in protected_regressions:
        if query_id in regressed_delta:
            protected_delta[query_id] = max(
                float(protected_delta.get(query_id, 0.0)),
                float(regressed_delta[query_id]),
            )
    protected_support: dict[str, dict[str, float]] = {}
    protected_observations: dict[str, dict[str, dict[str, Any]]] = {}
    protected_feedback_metadata: dict[str, Any] = {"baseline_feedback_bank": None}
    if baseline_feedback_bank is not None:
        protected_support, protected_observations, protected_feedback_metadata = _protected_support_from_feedback_bank(
            baseline_feedback_bank,
            protected_query_delta=protected_delta,
        )

    paired_deltas = [float(row["delta_sparse_te_cm"]) for row in paired if row.get("delta_sparse_te_cm") is not None]
    feedback_bank_count = int(baseline_feedback_bank is not None) + int(candidate_feedback_bank is not None)
    attribution_entry_count = int(len(landmark_risk)) + int(
        sum(len(query_map) for query_map in protected_support.values())
    ) + int(sum(len(query_map) for query_map in validated_support.values()))
    if feedback_bank_count <= 0:
        attribution_status = "metric_only"
    elif attribution_entry_count > 0:
        attribution_status = "attributed"
    else:
        attribution_status = "empty_attribution"
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v2",
        "scene": str(scene),
        "split_name": split,
        "source": "paired_sparse_pnp_validation",
        "attribution_status": attribution_status,
        "baseline_run_dir": str(Path(baseline_run_dir)),
        "candidate_run_dir": str(Path(candidate_run_dir)),
        "thresholds": {
            "protected_te_cm": float(protected_te_cm),
            "hard_te_cm": float(hard_te_cm),
            "regression_margin_cm": float(regression_margin_cm),
            "improvement_margin_cm": float(improvement_margin_cm),
            "protect_all_baseline_good_queries": bool(protect_all_baseline_good_queries),
        },
        "baseline_summary": _summary(baseline_rows),
        "candidate_summary": _summary(candidate_rows),
        "paired_queries": paired,
        "regression_query_ids": sorted(regression_queries),
        "protected_regression_query_ids": sorted(protected_regressions),
        "hard_improved_query_ids": sorted(hard_improvements),
        "query_hard_improvement_delta_cm": {
            query: float(delta) for query, delta in sorted(hard_improvement_delta.items())
        },
        "query_regression_delta_cm": {query: float(delta) for query, delta in sorted(regressed_delta.items())},
        "landmark_regression_risk": {gid: float(value) for gid, value in sorted(landmark_risk.items(), key=lambda item: int(item[0]))},
        "regression_per_query_risk_support": {
            query: {gid: float(value) for gid, value in sorted(query_map.items(), key=lambda item: int(item[0]))}
            for query, query_map in sorted(regression_per_query_risk.items())
        },
        "protected_per_query_support": {
            query: {gid: float(value) for gid, value in sorted(query_map.items(), key=lambda item: int(item[0]))}
            for query, query_map in sorted(protected_support.items())
        },
        "protected_per_query_observations": {
            query: {gid: dict(value) for gid, value in sorted(query_map.items(), key=lambda item: int(item[0]))}
            for query, query_map in sorted(protected_observations.items())
        },
        "validated_per_query_support": {
            query: {gid: float(value) for gid, value in sorted(query_map.items(), key=lambda item: int(item[0]))}
            for query, query_map in sorted(validated_support.items())
        },
        "validated_per_query_observations": {
            query: {gid: dict(value) for gid, value in sorted(query_map.items(), key=lambda item: int(item[0]))}
            for query, query_map in sorted(validated_observations.items())
        },
        "metrics": {
            "paired_query_count": int(len(paired)),
            "regression_20cm_count": int(sum(1 for delta in paired_deltas if delta >= 20.0)),
            "regression_50cm_count": int(sum(1 for delta in paired_deltas if delta >= 50.0)),
            "improvement_20cm_count": int(sum(1 for delta in paired_deltas if delta <= -20.0)),
            "protected_regression_count": int(len(protected_regressions)),
            "protected_query_count": int(len(protected_delta)),
            "validated_query_count": int(len(hard_improvement_delta)),
            "hard_improvement_count": int(len(hard_improvements)),
            "median_delta_sparse_te_cm": _median(paired_deltas),
            "p90_delta_sparse_te_cm": _percentile(paired_deltas, 0.90),
            "landmark_regression_risk_count": int(len(landmark_risk)),
            "regression_basin_landmark_support_count": int(
                sum(len(query_map) for query_map in regression_per_query_risk.values())
            ),
            "protected_landmark_support_count": int(sum(len(query_map) for query_map in protected_support.values())),
            "validated_landmark_support_count": int(sum(len(query_map) for query_map in validated_support.values())),
            "protected_observation_entry_count": int(
                sum(len(query_map) for query_map in protected_observations.values())
            ),
            "validated_observation_entry_count": int(
                sum(len(query_map) for query_map in validated_observations.values())
            ),
            "feedback_bank_count": int(feedback_bank_count),
            "attribution_entry_count": int(attribution_entry_count),
        },
        "feedback_metadata": feedback_metadata,
        "protected_feedback_metadata": protected_feedback_metadata,
        "validated_feedback_metadata": validated_feedback_metadata,
    }
    return profile


def sparse_validation_landmark_risk_tensor(
    profile: Mapping[str, Any],
    *,
    num_gaussians: int,
) -> torch.Tensor:
    split_name = str(profile.get("split_name", profile.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("test split sparse validation profile is not allowed")
    risk = torch.zeros(int(num_gaussians), dtype=torch.float32)
    raw = profile.get("landmark_regression_risk", {})
    if not isinstance(raw, Mapping):
        return risk
    for raw_gid, raw_value in raw.items():
        try:
            gid = int(raw_gid)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if 0 <= gid < int(num_gaussians) and math.isfinite(value) and value > 0.0:
            risk[gid] = max(float(risk[gid].item()), float(value))
    return risk
