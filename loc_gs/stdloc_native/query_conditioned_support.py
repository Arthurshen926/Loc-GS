from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import load_feedback_bank


def _load_payload(payload_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload_or_path, (str, Path)):
        payload = torch.load(Path(payload_or_path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("episode cache must contain a dict payload")
        return payload
    return dict(payload_or_path)


def _tensor(payload: dict[str, Any], *keys: str) -> torch.Tensor | None:
    for key in keys:
        if key in payload:
            return torch.as_tensor(payload[key]).cpu()
    return None


def _required_tensor(payload: dict[str, Any], *keys: str) -> torch.Tensor:
    value = _tensor(payload, *keys)
    if value is None:
        raise KeyError("/".join(keys))
    return value


def _source_set(source_idx: torch.Tensor | Any) -> set[int]:
    return {
        int(item)
        for item in torch.as_tensor(source_idx, dtype=torch.long).reshape(-1).cpu().tolist()
    }


def _map_landmark_ids(ids: torch.Tensor, base_gaussian_id: torch.Tensor | Any | None, num_gaussians: int) -> torch.Tensor:
    ids = ids.long().cpu()
    if base_gaussian_id is None:
        mapped = ids
    else:
        base = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(base.numel())):
            raise IndexError("candidate ids are outside base_gaussian_id")
        mapped = base[ids.clamp_min(0)]
    if mapped.numel() and (int(mapped.min()) < 0 or int(mapped.max()) >= int(num_gaussians)):
        raise IndexError("mapped candidate ids are outside num_gaussians")
    return mapped


def _positive_mask(
    payload: dict[str, Any],
    ids: torch.Tensor,
    *,
    reprojection_threshold_px: float,
    score_threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _tensor(payload, "candidate_cosine", "cosine")
    score = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score.shape != ids.shape:
        raise ValueError("candidate_cosine/cosine must match candidate landmark ids")
    visible = _tensor(payload, "candidate_visible", "visible")
    visible_mask = torch.ones_like(ids, dtype=torch.bool) if visible is None else visible.bool()
    if visible_mask.shape != ids.shape:
        raise ValueError("candidate_visible must match candidate landmark ids")
    pnp = _tensor(payload, "candidate_pnp_inlier", "pnp_inlier")
    pnp_mask = torch.ones_like(ids, dtype=torch.bool) if pnp is None else pnp.bool()
    if pnp_mask.shape != ids.shape:
        raise ValueError("candidate_pnp_inlier must match candidate landmark ids")
    labels = _tensor(payload, "pair_label")
    errors = _tensor(payload, "candidate_reprojection_error", "reprojection_error")
    if labels is not None:
        positive = labels.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
        quality = score.clamp_min(0.0)
    elif errors is not None:
        err = errors.float()
        if err.shape != ids.shape:
            raise ValueError("candidate_reprojection_error/reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(err)
        positive = finite & (err <= float(reprojection_threshold_px))
        quality = score.clamp_min(0.0)
    else:
        raise KeyError("pair_label or candidate_reprojection_error")
    if score_threshold is not None:
        positive = positive & (score >= float(score_threshold))
    return positive & visible_mask & pnp_mask, quality


def _empty_metrics() -> dict[str, float]:
    return {
        "support": 0.0,
        "viable_tuple_mass": 0.0,
        "logdet_H": 0.0,
        "min_eigenvalue": 0.0,
        "dense_worsen_risk": 0.0,
        "ambiguity": 0.0,
    }


def _add_metric(table: dict[str, dict[str, dict[str, float]]], gid: int, query_id: int, quality: float) -> None:
    per_landmark = table.setdefault(str(int(gid)), {})
    metrics = per_landmark.setdefault(str(int(query_id)), _empty_metrics())
    q = max(float(quality), 0.0)
    metrics["support"] += 1.0
    metrics["viable_tuple_mass"] += q
    metrics["logdet_H"] += q
    metrics["min_eigenvalue"] += q


def _add_metric_payload(
    table: dict[str, dict[str, dict[str, float]]],
    *,
    gid: int,
    query_id: str,
    metrics: dict[str, float],
) -> None:
    per_landmark = table.setdefault(str(int(gid)), {})
    current = per_landmark.setdefault(str(query_id), _empty_metrics())
    for key, value in metrics.items():
        current[key] += float(value)


def _hard_queries(query_ids: torch.Tensor, requested: Iterable[int] | None) -> list[int]:
    if requested is not None:
        return sorted({int(item) for item in requested})
    return sorted({int(item) for item in query_ids.reshape(-1).tolist()})


def build_query_conditioned_solver_constraints(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    source_idx: torch.Tensor | Any,
    num_gaussians: int,
    hard_query_ids: Iterable[int] | None = None,
    base_gaussian_id: torch.Tensor | Any | None = None,
    reprojection_threshold_px: float = 4.0,
    score_threshold: float | None = None,
    min_candidate_positive: float = 0.0,
    min_support_delta: float = 0.0,
    min_viable_tuple_delta: float = 0.0,
    min_logdet_delta: float = 0.0,
    max_ambiguity_delta: float = 0.0,
) -> dict[str, Any]:
    """Build query-conditioned replacement constraints from self-map feedback.

    The returned JSON-compatible payload is accepted by
    `export_lsf_solver_aware_map.py --solver_admissibility_path`.
    This function requires real per-row `query_id`; it intentionally does not
    synthesize query groups.
    """

    payload = _load_payload(payload_or_path)
    query_id = _required_tensor(payload, "query_id").long().reshape(-1).cpu()
    ids = _required_tensor(payload, "candidate_landmark_ids", "landmark_id").long().cpu()
    if ids.dim() != 2:
        raise ValueError("candidate_landmark_ids/landmark_id must have shape [N,K]")
    if query_id.numel() != ids.shape[0]:
        raise ValueError("query_id must have one value per candidate row")
    mapped = _map_landmark_ids(
        ids,
        payload.get("base_gaussian_id") if base_gaussian_id is None else base_gaussian_id,
        int(num_gaussians),
    )
    positive, quality = _positive_mask(
        payload,
        ids,
        reprojection_threshold_px=float(reprojection_threshold_px),
        score_threshold=score_threshold,
    )
    hard = _hard_queries(query_id, hard_query_ids)
    hard_set = set(hard)
    source = _source_set(source_idx)
    candidate_gain: dict[str, dict[str, dict[str, float]]] = {}
    source_loss: dict[str, dict[str, dict[str, float]]] = {}
    for row in range(int(ids.shape[0])):
        qid = int(query_id[row].item())
        if qid not in hard_set:
            continue
        cols = torch.where(positive[row])[0].tolist()
        for col in cols:
            gid = int(mapped[row, col].item())
            value = float(quality[row, col].item())
            if gid in source:
                _add_metric(source_loss, gid, qid, value)
            elif value >= float(min_candidate_positive):
                _add_metric(candidate_gain, gid, qid, value)
    metadata = dict(payload.get("metadata", {}))
    return {
        "format": "loc_gs_solver_admissibility_v1",
        "hard_query_ids": hard,
        "candidate_gain": candidate_gain,
        "source_loss": source_loss,
        "thresholds": {
            "min_support_delta": float(min_support_delta),
            "min_viable_tuple_delta": float(min_viable_tuple_delta),
            "min_logdet_delta": float(min_logdet_delta),
            "max_ambiguity_delta": float(max_ambiguity_delta),
        },
        "metadata": {
            "query_group_mode": "query_id",
            "query_count": int(len(set(query_id.tolist()))),
            "row_count": int(ids.shape[0]),
            "topk": int(ids.shape[1]),
            "source_count": int(len(source)),
            "candidate_gain_count": int(len(candidate_gain)),
            "source_loss_count": int(len(source_loss)),
            "split_name": str(metadata.get("split_name", metadata.get("feedback_bank_split", ""))),
            "reprojection_threshold_px": float(reprojection_threshold_px),
            "score_threshold": None if score_threshold is None else float(score_threshold),
            "min_candidate_positive": float(min_candidate_positive),
        },
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _feedback_gid(record: dict[str, Any]) -> int | None:
    for key in ("matched_gaussian_id", "matched_landmark_id", "gaussian_id", "landmark_id"):
        raw = record.get(key)
        if raw in (None, ""):
            continue
        try:
            gid = int(raw)
        except (TypeError, ValueError):
            continue
        return gid if gid >= 0 else None
    return None


def _feedback_group_key(record: dict[str, Any], group_field: str) -> str:
    if str(group_field) == "image_id":
        image_id = str(record.get("image_id", "")).strip()
        if image_id:
            return image_id
    query_id = str(record.get("query_id", "")).strip()
    return query_id.split("::", 1)[0] if query_id else "unknown"


def _feedback_positive(record: dict[str, Any], reprojection_threshold_px: float) -> bool:
    if not bool(record.get("pnp_inlier", False)):
        return False
    error = record.get("reprojection_error_px")
    if error is None:
        return True
    return _as_float(error, float("inf")) <= float(reprojection_threshold_px)


def _feedback_quality(record: dict[str, Any], reprojection_threshold_px: float) -> float:
    score = max(_as_float(record.get("descriptor_score"), 0.0), 0.0)
    error = record.get("reprojection_error_px")
    reproj_quality = 1.0 if error is None else max(0.0, 1.0 - (_as_float(error, float("inf")) / max(float(reprojection_threshold_px), 1e-6)))
    visibility = max(_as_float(record.get("visibility_score"), 1.0), 0.0)
    return float(score * reproj_quality * visibility)


def _feedback_dense_worsen_risk(record: dict[str, Any], dense_delta_bad_cm: float) -> float:
    transition = str(record.get("dense_transition", "")).strip().lower()
    risk = 1.0 if transition in {"worsened", "lost"} else 0.0
    delta = _as_float(record.get("dense_delta_te_cm"), 0.0)
    if delta > 0.0:
        risk = max(risk, min(1.0, delta / max(float(dense_delta_bad_cm), 1e-6)))
    return float(risk)


def _feedback_ambiguity(record: dict[str, Any], *, positive: bool, score_threshold: float, reprojection_threshold_px: float) -> float:
    if positive:
        return 0.0
    score = _as_float(record.get("descriptor_score"), 0.0)
    error = _as_float(record.get("reprojection_error_px"), 0.0)
    if score >= float(score_threshold) and error >= float(reprojection_threshold_px):
        return float(score)
    return 0.0


def _feedback_record_metrics(
    record: dict[str, Any],
    *,
    positive_reprojection_threshold_px: float,
    hard_negative_score_threshold: float,
    hard_negative_reprojection_threshold_px: float,
    dense_delta_bad_cm: float,
) -> tuple[dict[str, float], dict[str, float]]:
    positive = _feedback_positive(record, float(positive_reprojection_threshold_px))
    quality = _feedback_quality(record, float(positive_reprojection_threshold_px)) if positive else 0.0
    logdet = _as_float(record.get("jacobian_info_logdet_proxy"), math.log1p(max(quality, 0.0))) if positive else 0.0
    min_eigen = _as_float(record.get("jacobian_info_trace"), quality) if positive else 0.0
    dense_risk = _feedback_dense_worsen_risk(record, float(dense_delta_bad_cm))
    ambiguity = _feedback_ambiguity(
        record,
        positive=positive,
        score_threshold=float(hard_negative_score_threshold),
        reprojection_threshold_px=float(hard_negative_reprojection_threshold_px),
    )
    metrics = {
        "support": 1.0 if positive else 0.0,
        "viable_tuple_mass": float(quality),
        "logdet_H": float(logdet),
        "min_eigenvalue": float(min_eigen),
        "dense_worsen_risk": float(dense_risk),
        "ambiguity": float(ambiguity),
    }
    stats = {
        "support": metrics["support"],
        "viable_tuple_mass": metrics["viable_tuple_mass"],
        "dense_worsen_risk": metrics["dense_worsen_risk"],
        "ambiguity": metrics["ambiguity"],
        "pose_error_t_cm": _as_float(record.get("pose_error_t_cm"), 0.0),
        "pose_error_r_deg": _as_float(record.get("pose_error_r_deg"), 0.0),
    }
    return metrics, stats


def _select_feedback_hard_queries(
    query_stats: dict[str, dict[str, float]],
    *,
    topk: int,
    mode: str,
) -> list[str]:
    if int(topk) <= 0 or int(topk) >= len(query_stats):
        return sorted(query_stats)
    scored: list[tuple[float, str]] = []
    for query_id, stats in query_stats.items():
        if mode == "pose_error":
            score = float(stats.get("pose_error_t_cm", 0.0)) + 2.0 * float(stats.get("pose_error_r_deg", 0.0))
        elif mode == "dense_worsen":
            score = (
                2.0 * float(stats.get("dense_worsen_risk", 0.0))
                + 0.1 * float(stats.get("ambiguity", 0.0))
                + 0.02 * float(stats.get("pose_error_t_cm", 0.0))
            )
        else:
            score = (
                float(stats.get("ambiguity", 0.0))
                + 0.5 * float(stats.get("dense_worsen_risk", 0.0))
                - 0.02 * float(stats.get("support", 0.0))
            )
        scored.append((score, str(query_id)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(query_id for _, query_id in scored[: int(topk)])


def build_feedback_bank_v2_solver_constraints(
    feedback_bank: str | Path,
    *,
    source_idx: torch.Tensor | Any,
    num_gaussians: int | None = None,
    hard_query_ids: Iterable[str] | None = None,
    hard_query_topk: int = 64,
    hard_query_mode: str = "dense_worsen",
    group_field: str = "image_id",
    positive_reprojection_threshold_px: float = 4.0,
    hard_negative_score_threshold: float = 0.65,
    hard_negative_reprojection_threshold_px: float = 8.0,
    dense_delta_bad_cm: float = 5.0,
    min_candidate_positive: float = 0.0,
    min_support_delta: float = 0.0,
    min_viable_tuple_delta: float = 0.0,
    min_logdet_delta: float = 0.0,
    min_min_eigen_delta: float = 0.0,
    max_dense_worsen_delta: float = 0.0,
    max_ambiguity_delta: float = 0.0,
    cvar_alpha: float | None = 0.2,
    min_cvar_score: float = 0.0,
) -> dict[str, Any]:
    """Build solver admissibility constraints directly from audited feedback_bank_v2."""

    audit = audit_feedback_bank_v2(feedback_bank)
    if audit["audit_status"] != "passed":
        raise ValueError(f"feedback bank v2 audit failed: {audit['reasons']}")
    bank = load_feedback_bank(feedback_bank)
    manifest = dict(bank.get("manifest", {}))
    split_name = str(manifest.get("split_name", manifest.get("split", ""))).strip()
    if split_name.lower() == "test":
        raise ValueError("solver admissibility constraints cannot use test split")
    records = [dict(record) for record in bank.get("records", [])]
    source = _source_set(source_idx)
    inferred_count = max((_feedback_gid(record) or 0 for record in records), default=0) + 1
    if source:
        inferred_count = max(inferred_count, max(source) + 1)
    gaussian_count = inferred_count if num_gaussians is None else int(num_gaussians)
    if gaussian_count < inferred_count:
        raise ValueError("num_gaussians is smaller than feedback/source max gaussian id")

    query_stats: dict[str, dict[str, float]] = {}
    parsed: list[tuple[str, int, dict[str, float]]] = []
    for record in records:
        gid = _feedback_gid(record)
        if gid is None or gid >= gaussian_count:
            continue
        query_id = _feedback_group_key(record, str(group_field))
        metrics, stats = _feedback_record_metrics(
            record,
            positive_reprojection_threshold_px=float(positive_reprojection_threshold_px),
            hard_negative_score_threshold=float(hard_negative_score_threshold),
            hard_negative_reprojection_threshold_px=float(hard_negative_reprojection_threshold_px),
            dense_delta_bad_cm=float(dense_delta_bad_cm),
        )
        parsed.append((query_id, int(gid), metrics))
        aggregate = query_stats.setdefault(
            query_id,
            {
                "support": 0.0,
                "viable_tuple_mass": 0.0,
                "dense_worsen_risk": 0.0,
                "ambiguity": 0.0,
                "pose_error_t_cm": 0.0,
                "pose_error_r_deg": 0.0,
            },
        )
        for key in ("support", "viable_tuple_mass", "dense_worsen_risk", "ambiguity"):
            aggregate[key] += float(stats[key])
        aggregate["pose_error_t_cm"] = max(float(aggregate["pose_error_t_cm"]), float(stats["pose_error_t_cm"]))
        aggregate["pose_error_r_deg"] = max(float(aggregate["pose_error_r_deg"]), float(stats["pose_error_r_deg"]))

    hard = sorted({str(item) for item in hard_query_ids}) if hard_query_ids is not None else _select_feedback_hard_queries(
        query_stats,
        topk=int(hard_query_topk),
        mode=str(hard_query_mode),
    )
    hard_set = set(hard)
    candidate_gain: dict[str, dict[str, dict[str, float]]] = {}
    source_loss: dict[str, dict[str, dict[str, float]]] = {}
    for query_id, gid, metrics in parsed:
        if query_id not in hard_set:
            continue
        if gid in source:
            _add_metric_payload(source_loss, gid=gid, query_id=query_id, metrics=metrics)
        elif metrics["support"] > 0.0 and metrics["viable_tuple_mass"] >= float(min_candidate_positive):
            _add_metric_payload(candidate_gain, gid=gid, query_id=query_id, metrics=metrics)

    thresholds: dict[str, Any] = {
        "min_support_delta": float(min_support_delta),
        "min_viable_tuple_delta": float(min_viable_tuple_delta),
        "min_logdet_delta": float(min_logdet_delta),
        "min_min_eigen_delta": float(min_min_eigen_delta),
        "max_dense_worsen_delta": float(max_dense_worsen_delta),
        "max_ambiguity_delta": float(max_ambiguity_delta),
        "min_cvar_score": float(min_cvar_score),
        "cvar_weights": {
            "support": 1.0,
            "viable_tuple_mass": 1.0,
            "logdet_H": 1.0,
            "min_eigenvalue": 1.0,
            "dense_worsen_risk": -1.0,
            "ambiguity": -1.0,
        },
    }
    if cvar_alpha is not None:
        thresholds["cvar_alpha"] = float(cvar_alpha)
    return {
        "format": "loc_gs_solver_admissibility_v2_feedback_bank",
        "hard_query_ids": hard,
        "candidate_gain": candidate_gain,
        "source_loss": source_loss,
        "thresholds": thresholds,
        "metadata": {
            "query_group_mode": f"feedback_bank_v2:{group_field}",
            "query_count": int(len(query_stats)),
            "hard_query_count": int(len(hard)),
            "record_count": int(len(records)),
            "source_count": int(len(source)),
            "num_gaussians": int(gaussian_count),
            "candidate_gain_count": int(len(candidate_gain)),
            "source_loss_count": int(len(source_loss)),
            "split_name": split_name,
            "feedback_bank_schema": str(manifest.get("schema_version", "")),
            "feedback_bank": str(feedback_bank),
            "hard_query_mode": str(hard_query_mode),
            "hard_query_topk": int(hard_query_topk),
            "positive_reprojection_threshold_px": float(positive_reprojection_threshold_px),
            "hard_negative_score_threshold": float(hard_negative_score_threshold),
            "hard_negative_reprojection_threshold_px": float(hard_negative_reprojection_threshold_px),
            "dense_delta_bad_cm": float(dense_delta_bad_cm),
            "min_candidate_positive": float(min_candidate_positive),
            "split_audit": audit,
        },
    }
