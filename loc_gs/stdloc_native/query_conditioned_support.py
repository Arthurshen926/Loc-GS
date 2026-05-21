from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch


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
        "ambiguity": 0.0,
    }


def _add_metric(table: dict[str, dict[str, dict[str, float]]], gid: int, query_id: int, quality: float) -> None:
    per_landmark = table.setdefault(str(int(gid)), {})
    metrics = per_landmark.setdefault(str(int(query_id)), _empty_metrics())
    q = max(float(quality), 0.0)
    metrics["support"] += 1.0
    metrics["viable_tuple_mass"] += q
    metrics["logdet_H"] += q


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
