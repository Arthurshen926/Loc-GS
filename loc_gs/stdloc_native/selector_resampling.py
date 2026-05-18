from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from loc_gs.stdloc_native.soft_prior import _dump_pickle, _load_pickle


def _as_float_vector(values: torch.Tensor | Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 1:
        return torch.ones_like(values)
    order = torch.argsort(values, descending=False, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def _normalize_selector_for_sampling(selector: torch.Tensor, transform: str) -> torch.Tensor:
    transform = str(transform or "identity")
    if transform == "identity":
        return selector.clamp(0.0, 1.0)
    if transform == "rank":
        return _rank_normalize(selector)
    if transform == "minmax":
        lo = selector.min()
        hi = selector.max()
        if float((hi - lo).abs().item()) < 1e-8:
            return torch.full_like(selector, 0.5)
        return ((selector - lo) / (hi - lo)).clamp(0.0, 1.0)
    raise ValueError(f"unsupported selector_transform: {transform}")


def _cache_tensor(payload: dict[str, Any], *names: str) -> torch.Tensor | None:
    for name in names:
        if name in payload:
            return torch.as_tensor(payload[name])
    return None


def _cache_metadata(payload_or_path: dict[str, Any] | str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    if isinstance(payload_or_path, (str, Path)):
        metadata["cache_path"] = str(payload_or_path)
    return metadata


def hard_negative_risk_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate self-map high-score wrong matches into a per-Gaussian risk prior."""

    payload = (
        torch.load(Path(payload_or_path), map_location="cpu")
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    metadata = _cache_metadata(payload_or_path, payload)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    if errors is None and pair_label is None:
        raise KeyError("episode cache needs reprojection_error or pair_label to build hard-negative risk")
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        valid = valid & finite
    else:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
    if score_threshold is not None:
        hard_negative = valid & (~positive) & (score_tensor >= float(score_threshold))
    else:
        hard_negative = valid & (~positive)
    true_positive = valid & positive
    if base_gaussian_id is not None:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped_ids = base_ids[ids.clamp_min(0)]
    else:
        mapped_ids = ids
    num_gaussians = int(num_gaussians)
    if mapped_ids.numel() and (int(mapped_ids.min().item()) < 0 or int(mapped_ids.max().item()) >= num_gaussians):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    fp = torch.zeros(num_gaussians, dtype=torch.float32)
    tp = torch.zeros_like(fp)
    if hard_negative.any():
        fp.scatter_add_(0, mapped_ids[hard_negative].reshape(-1).cpu(), torch.ones(int(hard_negative.sum().item())))
    if true_positive.any():
        tp.scatter_add_(0, mapped_ids[true_positive].reshape(-1).cpu(), torch.ones(int(true_positive.sum().item())))
    denom = fp + tp
    risk = torch.zeros_like(fp)
    observed = denom > 0.0
    risk[observed] = fp[observed] / denom[observed].clamp_min(1e-8)
    summary = {
        "cache_path": metadata.get("cache_path", ""),
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "observed_landmarks": int(observed.sum().item()),
        "hard_negative_pairs": int(hard_negative.sum().item()),
        "true_positive_pairs": int(true_positive.sum().item()),
        "split_audit": metadata.get("split_audit", {}),
    }
    return risk.clamp(0.0, 1.0), summary


def positive_support_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate self-map correct matches into a normalized per-Gaussian support prior."""

    payload = (
        torch.load(Path(payload_or_path), map_location="cpu")
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    metadata = _cache_metadata(payload_or_path, payload)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    if errors is None and pair_label is None:
        raise KeyError("episode cache needs reprojection_error or pair_label to build positive support")
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        valid = valid & finite
    else:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
    if score_threshold is not None:
        positive = positive & (score_tensor >= float(score_threshold))
    positive = valid & positive
    if base_gaussian_id is not None:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped_ids = base_ids[ids.clamp_min(0)]
    else:
        mapped_ids = ids
    num_gaussians = int(num_gaussians)
    if mapped_ids.numel() and (int(mapped_ids.min().item()) < 0 or int(mapped_ids.max().item()) >= num_gaussians):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    counts = torch.zeros(num_gaussians, dtype=torch.float32)
    if positive.any():
        counts.scatter_add_(0, mapped_ids[positive].reshape(-1).cpu(), torch.ones(int(positive.sum().item())))
    support = torch.zeros_like(counts)
    max_count = float(counts.max().item()) if counts.numel() else 0.0
    if max_count > 0.0:
        support = counts / max_count
    positive_landmarks = counts > 0.0
    summary = {
        "cache_path": metadata.get("cache_path", ""),
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "positive_landmarks": int(positive_landmarks.sum().item()),
        "positive_pairs": int(positive.sum().item()),
        "max_positive_pairs_per_landmark": max_count,
        "split_audit": metadata.get("split_audit", {}),
    }
    return support.clamp(0.0, 1.0), summary


def pose_information_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate self-map inlier quality into a per-Gaussian pose-information prior."""

    payload = (
        torch.load(Path(payload_or_path), map_location="cpu")
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    metadata = _cache_metadata(payload_or_path, payload)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    if errors is None and pair_label is None:
        raise KeyError("episode cache needs reprojection_error or pair_label to build pose information")
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        valid = valid & finite
        reproj_quality = (1.0 - error_tensor / float(reprojection_threshold_px)).clamp(0.0, 1.0)
    else:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
        reproj_quality = torch.ones_like(score_tensor)
    if score_threshold is not None:
        positive = positive & (score_tensor >= float(score_threshold))
    positive = valid & positive
    query_score = _cache_tensor(payload, "query_score")
    if query_score is None:
        query_weight = torch.ones((ids.shape[0],), dtype=torch.float32)
    else:
        query_weight = torch.as_tensor(query_score, dtype=torch.float32).reshape(-1).cpu()
        if query_weight.shape[0] != ids.shape[0]:
            raise ValueError("query_score must have one value per query/candidate row")
        query_weight = query_weight.clamp_min(0.0)
    margin = _cache_tensor(payload, "margin")
    if margin is None:
        margin_weight = torch.ones((ids.shape[0],), dtype=torch.float32)
    else:
        margin_weight = torch.as_tensor(margin, dtype=torch.float32).reshape(-1).cpu()
        if margin_weight.shape[0] != ids.shape[0]:
            raise ValueError("margin must have one value per query/candidate row")
        margin_weight = margin_weight.clamp_min(0.0)
    weights = reproj_quality * query_weight[:, None] * margin_weight[:, None]
    weights = torch.where(positive, weights, torch.zeros_like(weights))
    if base_gaussian_id is not None:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped_ids = base_ids[ids.clamp_min(0)]
    else:
        mapped_ids = ids
    num_gaussians = int(num_gaussians)
    if mapped_ids.numel() and (int(mapped_ids.min().item()) < 0 or int(mapped_ids.max().item()) >= num_gaussians):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    info = torch.zeros(num_gaussians, dtype=torch.float32)
    if positive.any():
        info.scatter_add_(0, mapped_ids[positive].reshape(-1).cpu(), weights[positive].reshape(-1).cpu())
    max_info = float(info.max().item()) if info.numel() else 0.0
    normalized = torch.zeros_like(info)
    if max_info > 0.0:
        normalized = info / max_info
    pose_landmarks = info > 0.0
    summary = {
        "cache_path": metadata.get("cache_path", ""),
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "pose_landmarks": int(pose_landmarks.sum().item()),
        "positive_pairs": int(positive.sum().item()),
        "max_pose_information": max_info,
        "split_audit": metadata.get("split_audit", {}),
    }
    return normalized.clamp(0.0, 1.0), summary


def hard_query_support_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate correct self-map matches, upweighting low-margin ambiguous queries."""

    payload = (
        torch.load(Path(payload_or_path), map_location="cpu")
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    metadata = _cache_metadata(payload_or_path, payload)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    if errors is None and pair_label is None:
        raise KeyError("episode cache needs reprojection_error or pair_label to build hard-query support")
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        valid = valid & finite
        reproj_quality = (1.0 - error_tensor / float(reprojection_threshold_px)).clamp(0.0, 1.0)
    else:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
        reproj_quality = torch.ones_like(score_tensor)
    if score_threshold is not None:
        positive = positive & (score_tensor >= float(score_threshold))
    positive = valid & positive
    margin = _cache_tensor(payload, "margin")
    if margin is None:
        query_hardness = torch.ones((ids.shape[0],), dtype=torch.float32)
        margin_available = False
    else:
        margin_tensor = torch.as_tensor(margin, dtype=torch.float32).reshape(-1).cpu().clamp_min(0.0)
        if margin_tensor.shape[0] != ids.shape[0]:
            raise ValueError("margin must have one value per query/candidate row")
        max_margin = float(margin_tensor.max().item()) if margin_tensor.numel() else 0.0
        if max_margin > 0.0:
            query_hardness = (1.0 - margin_tensor / max_margin).clamp(0.0, 1.0)
        else:
            query_hardness = torch.ones_like(margin_tensor)
        margin_available = True
    weights = reproj_quality * (1.0 + query_hardness[:, None])
    weights = torch.where(positive, weights, torch.zeros_like(weights))
    if base_gaussian_id is not None:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped_ids = base_ids[ids.clamp_min(0)]
    else:
        mapped_ids = ids
    num_gaussians = int(num_gaussians)
    if mapped_ids.numel() and (int(mapped_ids.min().item()) < 0 or int(mapped_ids.max().item()) >= num_gaussians):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    support = torch.zeros(num_gaussians, dtype=torch.float32)
    if positive.any():
        support.scatter_add_(0, mapped_ids[positive].reshape(-1).cpu(), weights[positive].reshape(-1).cpu())
    max_support = float(support.max().item()) if support.numel() else 0.0
    normalized = torch.zeros_like(support)
    if max_support > 0.0:
        normalized = support / max_support
    hard_query_landmarks = support > 0.0
    summary = {
        "cache_path": metadata.get("cache_path", ""),
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "positive_pairs": int(positive.sum().item()),
        "hard_query_landmarks": int(hard_query_landmarks.sum().item()),
        "max_hard_query_support": max_support,
        "margin_available": bool(margin_available),
        "split_audit": metadata.get("split_audit", {}),
    }
    return normalized.clamp(0.0, 1.0), summary


def query_coverage_reservation_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
    margin_threshold: float | None = None,
    hard_query_fraction: float = 0.25,
    max_landmarks: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedily order landmarks that cover low-margin self-map inlier queries."""

    payload = (
        torch.load(Path(payload_or_path), map_location="cpu")
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    metadata = _cache_metadata(payload_or_path, payload)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    if errors is None and pair_label is None:
        raise KeyError("episode cache needs reprojection_error or pair_label to build query coverage")
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        valid = valid & finite
        reproj_quality = (1.0 - error_tensor / float(reprojection_threshold_px)).clamp(0.0, 1.0)
    else:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
        reproj_quality = torch.ones_like(score_tensor)
    if score_threshold is not None:
        positive = positive & (score_tensor >= float(score_threshold))
    positive = valid & positive
    margin = _cache_tensor(payload, "margin")
    selected_margin_threshold: float | None = None
    if margin is None:
        hard_queries = positive.any(dim=1)
    else:
        margin_tensor = torch.as_tensor(margin, dtype=torch.float32).reshape(-1).cpu()
        if margin_tensor.shape[0] != ids.shape[0]:
            raise ValueError("margin must have one value per query/candidate row")
        if margin_threshold is not None:
            selected_margin_threshold = float(margin_threshold)
        else:
            fraction = min(max(float(hard_query_fraction), 0.0), 1.0)
            if fraction <= 0.0:
                selected_margin_threshold = float("-inf")
            else:
                order = torch.sort(margin_tensor).values
                kth = min(max(int(round(float(order.numel()) * fraction)) - 1, 0), int(order.numel()) - 1)
                selected_margin_threshold = float(order[kth].item())
        hard_queries = (margin_tensor <= selected_margin_threshold) & positive.any(dim=1)
    if base_gaussian_id is not None:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped_ids = base_ids[ids.clamp_min(0)]
    else:
        mapped_ids = ids
    num_gaussians = int(num_gaussians)
    if mapped_ids.numel() and (int(mapped_ids.min().item()) < 0 or int(mapped_ids.max().item()) >= num_gaussians):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    hard_positive = positive & hard_queries[:, None]
    row_ids, col_ids = torch.where(hard_positive)
    selected: list[int] = []
    covered: set[int] = set()
    candidate_rows: dict[int, set[int]] = {}
    candidate_weight: dict[int, float] = {}
    for row, col in zip(row_ids.tolist(), col_ids.tolist()):
        gid = int(mapped_ids[row, col].item())
        candidate_rows.setdefault(gid, set()).add(int(row))
        weight = float((reproj_quality[row, col] * score_tensor[row, col]).item())
        candidate_weight[gid] = candidate_weight.get(gid, 0.0) + weight
    limit = len(candidate_rows) if max_landmarks is None else max(0, int(max_landmarks))
    while len(selected) < limit and len(selected) < len(candidate_rows):
        best_gid: int | None = None
        best_key: tuple[int, float, int] | None = None
        for gid, rows in candidate_rows.items():
            if gid in selected:
                continue
            new_cover = len(rows - covered)
            if new_cover > 0:
                key = (new_cover, candidate_weight.get(gid, 0.0), -gid)
            else:
                key = (0, candidate_weight.get(gid, 0.0), -gid)
            if best_key is None or key > best_key:
                best_key = key
                best_gid = gid
        if best_gid is None:
            break
        selected.append(best_gid)
        covered.update(candidate_rows[best_gid])
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    summary = {
        "cache_path": metadata.get("cache_path", ""),
        "feedback_bank_split_name": metadata.get("feedback_bank_split_name", metadata.get("split_name", "")),
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "margin_threshold": selected_margin_threshold,
        "hard_query_fraction": float(hard_query_fraction),
        "hard_query_count": int(hard_queries.sum().item()),
        "covered_query_count": int(len(covered)),
        "candidate_landmarks": int(len(candidate_rows)),
        "selected_count": int(selected_tensor.numel()),
        "split_audit": metadata.get("split_audit", {}),
    }
    return selected_tensor, summary


def _support_rows_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
    margin_threshold: float | None = None,
    hard_query_fraction: float = 0.25,
) -> dict[str, Any]:
    payload = (
        torch.load(Path(payload_or_path), map_location="cpu")
        if isinstance(payload_or_path, (str, Path))
        else dict(payload_or_path)
    )
    metadata = _cache_metadata(payload_or_path, payload)
    ids = _cache_tensor(payload, "candidate_landmark_ids", "landmark_id")
    if ids is None:
        raise KeyError("episode cache is missing candidate_landmark_ids or landmark_id")
    ids = ids.long()
    if ids.dim() != 2:
        raise ValueError("candidate landmark ids must have shape [N,K]")
    mask = _cache_tensor(payload, "candidate_mask")
    valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
    if valid.shape != ids.shape:
        raise ValueError("candidate_mask must match candidate landmark ids")
    scores = _cache_tensor(payload, "candidate_cosine", "cosine")
    score_tensor = torch.ones_like(ids, dtype=torch.float32) if scores is None else scores.float()
    if score_tensor.shape != ids.shape:
        raise ValueError("candidate scores must match candidate landmark ids")
    errors = _cache_tensor(payload, "candidate_reprojection_error", "reprojection_error")
    pair_label = _cache_tensor(payload, "pair_label")
    if errors is None and pair_label is None:
        raise KeyError("episode cache needs reprojection_error or pair_label to build support constraints")
    if reprojection_threshold_px is None:
        reprojection_threshold_px = float(metadata.get("reprojection_threshold_px", 8.0))
    if errors is not None:
        error_tensor = errors.float()
        if error_tensor.shape != ids.shape:
            raise ValueError("reprojection_error must match candidate landmark ids")
        finite = torch.isfinite(error_tensor)
        positive = finite & (error_tensor <= float(reprojection_threshold_px))
        valid = valid & finite
        reproj_quality = (1.0 - error_tensor / float(reprojection_threshold_px)).clamp(0.0, 1.0)
    else:
        positive = pair_label.bool()
        if positive.shape != ids.shape:
            raise ValueError("pair_label must match candidate landmark ids")
        reproj_quality = torch.ones_like(score_tensor)
    if score_threshold is not None:
        positive = positive & (score_tensor >= float(score_threshold))
    positive = valid & positive
    margin = _cache_tensor(payload, "margin")
    selected_margin_threshold: float | None = None
    if margin is None:
        hard_queries = positive.any(dim=1)
    else:
        margin_tensor = torch.as_tensor(margin, dtype=torch.float32).reshape(-1).cpu()
        if margin_tensor.shape[0] != ids.shape[0]:
            raise ValueError("margin must have one value per query/candidate row")
        if margin_threshold is not None:
            selected_margin_threshold = float(margin_threshold)
        else:
            fraction = min(max(float(hard_query_fraction), 0.0), 1.0)
            if fraction <= 0.0:
                selected_margin_threshold = float("-inf")
            else:
                order = torch.sort(margin_tensor).values
                kth = min(max(int(round(float(order.numel()) * fraction)) - 1, 0), int(order.numel()) - 1)
                selected_margin_threshold = float(order[kth].item())
        hard_queries = (margin_tensor <= selected_margin_threshold) & positive.any(dim=1)
    if base_gaussian_id is not None:
        base_ids = torch.as_tensor(base_gaussian_id, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(base_ids.numel())):
            raise IndexError("episode cache landmark ids are outside base_gaussian_id")
        mapped_ids = base_ids[ids.clamp_min(0)]
    else:
        mapped_ids = ids
    num_gaussians = int(num_gaussians)
    if mapped_ids.numel() and (int(mapped_ids.min().item()) < 0 or int(mapped_ids.max().item()) >= num_gaussians):
        raise IndexError("mapped landmark ids are outside num_gaussians")
    return {
        "mapped_ids": mapped_ids,
        "positive": positive,
        "weights": reproj_quality * score_tensor.clamp_min(0.0),
        "hard_queries": hard_queries,
        "margin_threshold": selected_margin_threshold,
        "metadata": metadata,
        "score_threshold": score_threshold,
        "reprojection_threshold_px": float(reprojection_threshold_px),
        "hard_query_fraction": float(hard_query_fraction),
    }


def native_support_reservation_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    source_idx: torch.Tensor | Any,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
    margin_threshold: float | None = None,
    hard_query_fraction: float = 0.25,
    max_landmarks: int | None = None,
    preserve_counts: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedily reserve native landmarks that cover low-margin self-map queries."""

    rows = _support_rows_from_episode_cache(
        payload_or_path,
        num_gaussians=num_gaussians,
        base_gaussian_id=base_gaussian_id,
        score_threshold=score_threshold,
        reprojection_threshold_px=reprojection_threshold_px,
        margin_threshold=margin_threshold,
        hard_query_fraction=hard_query_fraction,
    )
    mapped_ids: torch.Tensor = rows["mapped_ids"]
    positive: torch.Tensor = rows["positive"]
    weights: torch.Tensor = rows["weights"]
    hard_queries: torch.Tensor = rows["hard_queries"]
    source = torch.as_tensor(source_idx, dtype=torch.long).reshape(-1).cpu()
    source_set = {int(idx) for idx in source.tolist() if 0 <= int(idx) < int(num_gaussians)}
    candidate_rows: dict[int, set[int]] = {}
    candidate_weight: dict[int, float] = {}
    hard_row_ids = torch.where(hard_queries)[0].tolist()
    for row in hard_row_ids:
        cols = torch.where(positive[row])[0].tolist()
        for col in cols:
            gid = int(mapped_ids[row, col].item())
            if gid not in source_set:
                continue
            candidate_rows.setdefault(gid, set()).add(int(row))
            candidate_weight[gid] = candidate_weight.get(gid, 0.0) + float(weights[row, col].item())
    limit = len(candidate_rows) if max_landmarks is None else max(0, int(max_landmarks))
    selected: list[int] = []
    covered: set[int] = set()
    while len(selected) < limit and len(selected) < len(candidate_rows):
        best_gid: int | None = None
        best_key: tuple[int, float, int] | None = None
        for gid, query_rows in candidate_rows.items():
            if gid in selected:
                continue
            new_cover = len(query_rows - covered)
            key = (new_cover, candidate_weight.get(gid, 0.0), -gid)
            if best_key is None or key > best_key:
                best_key = key
                best_gid = gid
        if best_gid is None or best_key is None or best_key[0] <= 0:
            break
        selected.append(best_gid)
        covered.update(candidate_rows[best_gid])
    if bool(preserve_counts) and len(selected) < limit:
        remaining = [
            gid
            for gid, _rows in sorted(
                candidate_rows.items(),
                key=lambda item: (-candidate_weight.get(item[0], 0.0), item[0]),
            )
            if gid not in selected
        ]
        for gid in remaining:
            if len(selected) >= limit:
                break
            selected.append(gid)
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    metadata = rows["metadata"]
    source_supported_rows = {
        row for query_rows in candidate_rows.values() for row in query_rows
    }
    summary = {
        "cache_path": metadata.get("cache_path", ""),
        "feedback_bank_split_name": metadata.get("feedback_bank_split_name", metadata.get("split_name", "")),
        "score_threshold": rows["score_threshold"],
        "reprojection_threshold_px": rows["reprojection_threshold_px"],
        "margin_threshold": rows["margin_threshold"],
        "hard_query_fraction": rows["hard_query_fraction"],
        "hard_query_count": int(hard_queries.sum().item()),
        "hard_query_with_source_support_count": int(len(source_supported_rows)),
        "covered_query_count": int(len(covered)),
        "source_support_landmarks": int(len(candidate_rows)),
        "selected_count": int(selected_tensor.numel()),
        "preserve_counts": bool(preserve_counts),
        "split_audit": metadata.get("split_audit", {}),
    }
    return selected_tensor, summary


def support_preservation_audit_from_episode_cache(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    source_idx: torch.Tensor | Any,
    candidate_idx: torch.Tensor | Any,
    num_gaussians: int,
    base_gaussian_id: torch.Tensor | Any | None = None,
    score_threshold: float | None = 0.0,
    reprojection_threshold_px: float | None = None,
    margin_threshold: float | None = None,
    hard_query_fraction: float = 0.25,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Summarize whether a candidate sampled set preserves native query support."""

    rows = _support_rows_from_episode_cache(
        payload_or_path,
        num_gaussians=num_gaussians,
        base_gaussian_id=base_gaussian_id,
        score_threshold=score_threshold,
        reprojection_threshold_px=reprojection_threshold_px,
        margin_threshold=margin_threshold,
        hard_query_fraction=hard_query_fraction,
    )
    mapped_ids: torch.Tensor = rows["mapped_ids"]
    positive: torch.Tensor = rows["positive"]
    hard_queries: torch.Tensor = rows["hard_queries"]
    source = torch.as_tensor(source_idx, dtype=torch.long).reshape(-1).cpu()
    candidate = torch.as_tensor(candidate_idx, dtype=torch.long).reshape(-1).cpu()
    source_set = {int(idx) for idx in source.tolist() if 0 <= int(idx) < int(num_gaussians)}
    candidate_set = {int(idx) for idx in candidate.tolist() if 0 <= int(idx) < int(num_gaussians)}
    source_counts: list[int] = []
    candidate_counts: list[int] = []
    source_support_landmarks: set[int] = set()
    candidate_support_landmarks: set[int] = set()
    examples: list[dict[str, int]] = []
    for row in range(int(mapped_ids.shape[0])):
        ids = {int(mapped_ids[row, col].item()) for col in torch.where(positive[row])[0].tolist()}
        native_ids = ids & source_set
        sampled_ids = ids & candidate_set
        source_counts.append(len(native_ids))
        candidate_counts.append(len(sampled_ids))
        if bool(hard_queries[row].item()):
            source_support_landmarks.update(native_ids)
            candidate_support_landmarks.update(sampled_ids)
        delta = len(sampled_ids) - len(native_ids)
        if delta < 0:
            examples.append(
                {
                    "query_index": int(row),
                    "source_support": int(len(native_ids)),
                    "candidate_support": int(len(sampled_ids)),
                    "support_delta": int(delta),
                    "hard_query": int(bool(hard_queries[row].item())),
                }
            )
    source_tensor = torch.tensor(source_counts, dtype=torch.int64)
    candidate_tensor = torch.tensor(candidate_counts, dtype=torch.int64)
    delta_tensor = candidate_tensor - source_tensor
    active = source_tensor > 0
    hard_active = active & hard_queries
    losses = active & (delta_tensor < 0)
    hard_losses = hard_active & (delta_tensor < 0)
    metadata = rows["metadata"]
    worst_delta = int(delta_tensor[active].min().item()) if bool(active.any().item()) else 0
    hard_worst_delta = int(delta_tensor[hard_active].min().item()) if bool(hard_active.any().item()) else 0
    examples = sorted(examples, key=lambda item: (item["support_delta"], -item["source_support"], item["query_index"]))
    return {
        "cache_path": metadata.get("cache_path", ""),
        "feedback_bank_split_name": metadata.get("feedback_bank_split_name", metadata.get("split_name", "")),
        "score_threshold": rows["score_threshold"],
        "reprojection_threshold_px": rows["reprojection_threshold_px"],
        "margin_threshold": rows["margin_threshold"],
        "hard_query_fraction": rows["hard_query_fraction"],
        "query_count": int(mapped_ids.shape[0]),
        "supported_query_count": int(active.sum().item()),
        "hard_query_count": int(hard_queries.sum().item()),
        "hard_supported_query_count": int(hard_active.sum().item()),
        "query_loss_count": int(losses.sum().item()),
        "hard_query_loss_count": int(hard_losses.sum().item()),
        "worst_support_delta": worst_delta,
        "hard_query_worst_support_delta": hard_worst_delta,
        "mean_support_delta": float(delta_tensor[active].float().mean().item()) if bool(active.any().item()) else 0.0,
        "source_support_landmarks": int(len(source_support_landmarks)),
        "candidate_support_landmarks": int(len(candidate_support_landmarks)),
        "source_sampled_count": int(source.numel()),
        "candidate_sampled_count": int(candidate.numel()),
        "native_kept_count": int(len(source_set & candidate_set)),
        "native_dropped_count": int(len(source_set - candidate_set)),
        "added_non_native_count": int(len(candidate_set - source_set)),
        "loss_examples": examples[: max(0, int(max_examples))],
        "split_audit": metadata.get("split_audit", {}),
    }


def _load_source_scores(source_detector_dir: Path, *, num_gaussians: int) -> torch.Tensor:
    scores_path = source_detector_dir / "sampled_scores.pkl"
    sampled_idx_path = source_detector_dir / "sampled_idx.pkl"
    if not scores_path.exists() or not sampled_idx_path.exists():
        return torch.zeros(num_gaussians, dtype=torch.float32)
    sampled_idx = torch.as_tensor(_load_pickle(sampled_idx_path), dtype=torch.long).reshape(-1).cpu()
    payload = _load_pickle(scores_path)
    if isinstance(payload, dict) and "score_avg" in payload:
        score_avg = torch.as_tensor(payload["score_avg"], dtype=torch.float32).reshape(-1).cpu()
        if score_avg.shape[0] == num_gaussians:
            return score_avg
    sampled_scores = (
        torch.as_tensor(payload.get("sampled_scores"), dtype=torch.float32).reshape(-1).cpu()
        if isinstance(payload, dict) and "sampled_scores" in payload
        else torch.as_tensor(payload, dtype=torch.float32).reshape(-1).cpu()
    )
    full = torch.zeros(num_gaussians, dtype=torch.float32)
    if sampled_scores.shape[0] == sampled_idx.shape[0]:
        valid = (sampled_idx >= 0) & (sampled_idx < num_gaussians)
        full[sampled_idx[valid]] = sampled_scores[valid]
    return full


def build_selector_sampling_scores(
    *,
    selector: torch.Tensor | Any,
    source_scores: torch.Tensor | Any | None = None,
    selector_weight: float = 1.0,
    source_score_weight: float = 1.0,
    selector_transform: str = "identity",
    hard_negative_risk: torch.Tensor | Any | None = None,
    hard_negative_weight: float = 0.0,
    positive_support: torch.Tensor | Any | None = None,
    positive_support_weight: float = 0.0,
    pose_information: torch.Tensor | Any | None = None,
    pose_information_weight: float = 0.0,
    hard_query_support: torch.Tensor | Any | None = None,
    hard_query_support_weight: float = 0.0,
) -> torch.Tensor:
    """Combine native detector scores with a learned localization selector."""

    selector_tensor = _normalize_selector_for_sampling(
        _as_float_vector(selector, name="selector"),
        str(selector_transform),
    )
    if source_scores is None:
        source_tensor = torch.zeros_like(selector_tensor)
    else:
        source_tensor = _as_float_vector(source_scores, name="source_scores")
        if source_tensor.shape[0] != selector_tensor.shape[0]:
            raise ValueError("source_scores must have the same length as selector")
    scores = (
        float(selector_weight) * selector_tensor
        + float(source_score_weight) * source_tensor.clamp(0.0, 1.0)
    )
    if hard_negative_risk is not None and float(hard_negative_weight) != 0.0:
        risk_tensor = _as_float_vector(hard_negative_risk, name="hard_negative_risk")
        if risk_tensor.shape[0] != selector_tensor.shape[0]:
            raise ValueError("hard_negative_risk must have the same length as selector")
        scores = scores - float(hard_negative_weight) * risk_tensor.clamp(0.0, 1.0)
    if positive_support is not None and float(positive_support_weight) != 0.0:
        support_tensor = _as_float_vector(positive_support, name="positive_support")
        if support_tensor.shape[0] != selector_tensor.shape[0]:
            raise ValueError("positive_support must have the same length as selector")
        scores = scores + float(positive_support_weight) * support_tensor.clamp(0.0, 1.0)
    if pose_information is not None and float(pose_information_weight) != 0.0:
        pose_tensor = _as_float_vector(pose_information, name="pose_information")
        if pose_tensor.shape[0] != selector_tensor.shape[0]:
            raise ValueError("pose_information must have the same length as selector")
        scores = scores + float(pose_information_weight) * pose_tensor.clamp(0.0, 1.0)
    if hard_query_support is not None and float(hard_query_support_weight) != 0.0:
        hard_query_tensor = _as_float_vector(hard_query_support, name="hard_query_support")
        if hard_query_tensor.shape[0] != selector_tensor.shape[0]:
            raise ValueError("hard_query_support must have the same length as selector")
        scores = scores + float(hard_query_support_weight) * hard_query_tensor.clamp(0.0, 1.0)
    return scores


def _coverage_cells(xyz: torch.Tensor, coverage_grid: int) -> torch.Tensor:
    if int(coverage_grid) <= 0:
        return torch.zeros((xyz.shape[0],), dtype=torch.long)
    xyz = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3).cpu()
    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    grid = int(coverage_grid)
    coords = (((xyz - mins) / span) * grid).floor().long().clamp(0, grid - 1)
    return coords[:, 0] * grid * grid + coords[:, 1] * grid + coords[:, 2]


def coverage_constrained_topk(
    *,
    scores: torch.Tensor | Any,
    xyz: torch.Tensor | Any,
    budget: int,
    coverage_grid: int = 0,
) -> torch.Tensor:
    """Select top scoring landmarks while giving each occupied cell a first pass."""

    score_tensor = _as_float_vector(scores, name="scores")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    budget = min(budget, score_tensor.numel())
    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3).cpu()
    if xyz_tensor.shape[0] != score_tensor.shape[0]:
        raise ValueError("xyz must have one row per score")
    order = torch.argsort(score_tensor, descending=True, stable=True)
    if int(coverage_grid) <= 0:
        return order[:budget].long()
    cells = _coverage_cells(xyz_tensor, int(coverage_grid))
    selected: list[int] = []
    seen_cells: set[int] = set()
    for idx in order.tolist():
        cell = int(cells[idx].item())
        if cell in seen_cells:
            continue
        selected.append(int(idx))
        seen_cells.add(cell)
        if len(selected) == budget:
            return torch.tensor(selected, dtype=torch.long)
    for idx in order.tolist():
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) == budget:
            break
    return torch.tensor(selected, dtype=torch.long)


def _topk_from_candidates(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    xyz: torch.Tensor,
    budget: int,
    coverage_grid: int,
    exclude: set[int] | None = None,
) -> torch.Tensor:
    if int(budget) <= 0 or candidate_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    if exclude:
        keep = torch.tensor([int(idx) not in exclude for idx in candidate_ids.tolist()], dtype=torch.bool)
        candidate_ids = candidate_ids[keep]
    if candidate_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    selected_local = coverage_constrained_topk(
        scores=scores[candidate_ids],
        xyz=xyz[candidate_ids],
        budget=min(int(budget), int(candidate_ids.numel())),
        coverage_grid=int(coverage_grid),
    )
    return candidate_ids[selected_local]


def _ordered_unique_candidate_ids(
    values: torch.Tensor | Any,
    *,
    candidate_ids: torch.Tensor,
    num_gaussians: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    raw = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    candidate_set = set(int(idx) for idx in candidate_ids.tolist())
    selected: list[int] = []
    seen: set[int] = set()
    in_range_count = 0
    for value in raw.tolist():
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        if 0 <= idx < int(num_gaussians):
            in_range_count += 1
        if 0 <= idx < int(num_gaussians) and idx in candidate_set:
            selected.append(idx)
    metadata = {
        "input_count": int(raw.numel()),
        "unique_count": int(len(seen)),
        "in_range_count": int(in_range_count),
        "candidate_count": int(len(selected)),
    }
    return torch.tensor(selected, dtype=torch.long), metadata


def resample_detector_landmarks(
    *,
    source_detector_dir: str | Path,
    selector: torch.Tensor | Any,
    xyz: torch.Tensor | Any,
    budget: str | int = "same_as_source",
    selector_weight: float = 1.0,
    source_score_weight: float = 1.0,
    selector_transform: str = "identity",
    coverage_grid: int = 0,
    candidate_pool: str = "all_gaussians",
    preserve_source_order: bool = False,
    source_retention_fraction: float = 0.0,
    hard_negative_risk: torch.Tensor | Any | None = None,
    hard_negative_weight: float = 0.0,
    hard_negative_metadata: dict[str, Any] | None = None,
    positive_support: torch.Tensor | Any | None = None,
    positive_support_weight: float = 0.0,
    positive_support_metadata: dict[str, Any] | None = None,
    pose_information: torch.Tensor | Any | None = None,
    pose_information_weight: float = 0.0,
    pose_information_metadata: dict[str, Any] | None = None,
    hard_query_support: torch.Tensor | Any | None = None,
    hard_query_support_weight: float = 0.0,
    hard_query_support_metadata: dict[str, Any] | None = None,
    support_guard_idx: torch.Tensor | Any | None = None,
    support_guard_fraction: float = 0.0,
    support_guard_metadata: dict[str, Any] | None = None,
    query_coverage_idx: torch.Tensor | Any | None = None,
    query_coverage_fraction: float = 0.0,
    query_coverage_metadata: dict[str, Any] | None = None,
    strict_support: torch.Tensor | Any | None = None,
    strict_support_fraction: float = 0.0,
    strict_support_metadata: dict[str, Any] | None = None,
    protected_source_idx: torch.Tensor | Any | None = None,
    protected_source_fraction: float = 1.0,
    protected_source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a resampled STDLoc detector payload without writing it."""

    source_detector_dir = Path(source_detector_dir)
    sampled_idx_path = source_detector_dir / "sampled_idx.pkl"
    if not sampled_idx_path.exists():
        raise FileNotFoundError(f"missing sampled_idx.pkl: {sampled_idx_path}")
    source_idx = torch.as_tensor(_load_pickle(sampled_idx_path), dtype=torch.long).reshape(-1).cpu()
    selector_tensor = _as_float_vector(selector, name="selector").clamp(0.0, 1.0)
    source_scores = _load_source_scores(source_detector_dir, num_gaussians=int(selector_tensor.shape[0]))
    scores = build_selector_sampling_scores(
        selector=selector_tensor,
        source_scores=source_scores,
        selector_weight=float(selector_weight),
        source_score_weight=float(source_score_weight),
        selector_transform=str(selector_transform),
        hard_negative_risk=hard_negative_risk,
        hard_negative_weight=float(hard_negative_weight),
        positive_support=positive_support,
        positive_support_weight=float(positive_support_weight),
        pose_information=pose_information,
        pose_information_weight=float(pose_information_weight),
        hard_query_support=hard_query_support,
        hard_query_support_weight=float(hard_query_support_weight),
    )
    candidate_pool = str(candidate_pool)
    if candidate_pool not in {"all_gaussians", "source_sampled"}:
        raise ValueError(f"unsupported candidate_pool: {candidate_pool}")
    if candidate_pool == "source_sampled":
        valid = (source_idx >= 0) & (source_idx < selector_tensor.shape[0])
        candidate_ids = source_idx[valid]
    else:
        candidate_ids = torch.arange(selector_tensor.shape[0], dtype=torch.long)
    output_budget = int(source_idx.numel()) if str(budget) == "same_as_source" else int(budget)
    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32).reshape(-1, 3).cpu()
    strict_fraction = min(max(float(strict_support_fraction), 0.0), 1.0)
    strict_selected = torch.empty(0, dtype=torch.long)
    strict_tensor: torch.Tensor | None = None
    if strict_support is not None:
        strict_tensor = _as_float_vector(strict_support, name="strict_support")
        if strict_tensor.shape[0] != selector_tensor.shape[0]:
            raise ValueError("strict_support must have the same length as selector")
    if strict_tensor is not None and strict_fraction > 0.0:
        strict_budget = min(int(round(float(output_budget) * strict_fraction)), int(output_budget))
        strict_candidates = candidate_ids[strict_tensor[candidate_ids] > 0.0]
        strict_scores = strict_tensor + 1e-4 * scores.clamp(0.0, 1.0)
        strict_selected = _topk_from_candidates(
            candidate_ids=strict_candidates,
            scores=strict_scores,
            xyz=xyz_tensor,
            budget=strict_budget,
            coverage_grid=int(coverage_grid),
        )
    support_guard_fraction = min(max(float(support_guard_fraction), 0.0), 1.0)
    support_guard_selected = torch.empty(0, dtype=torch.long)
    support_guard_metadata_out = dict(support_guard_metadata or {})
    if support_guard_idx is not None:
        support_candidates, support_counts = _ordered_unique_candidate_ids(
            support_guard_idx,
            candidate_ids=candidate_ids,
            num_gaussians=int(selector_tensor.shape[0]),
        )
        support_guard_metadata_out.update(support_counts)
        if support_guard_fraction > 0.0:
            strict_exclude = set(int(item) for item in strict_selected.tolist())
            if strict_exclude:
                keep = torch.tensor(
                    [int(idx) not in strict_exclude for idx in support_candidates.tolist()],
                    dtype=torch.bool,
                )
                support_candidates = support_candidates[keep]
            support_guard_metadata_out["available_count"] = int(support_candidates.numel())
            support_budget = min(
                int(round(float(output_budget) * support_guard_fraction)),
                int(output_budget) - int(strict_selected.numel()),
                int(support_candidates.numel()),
            )
            if support_budget > 0:
                support_guard_selected = support_candidates[:support_budget].long()
    query_coverage_fraction = min(max(float(query_coverage_fraction), 0.0), 1.0)
    query_coverage_selected = torch.empty(0, dtype=torch.long)
    query_coverage_metadata_out = dict(query_coverage_metadata or {})
    if query_coverage_idx is not None:
        query_candidates, query_counts = _ordered_unique_candidate_ids(
            query_coverage_idx,
            candidate_ids=candidate_ids,
            num_gaussians=int(selector_tensor.shape[0]),
        )
        query_coverage_metadata_out.update(query_counts)
        if query_coverage_fraction > 0.0:
            reserved_exclude = set(int(item) for item in torch.cat([strict_selected, support_guard_selected]).tolist())
            if reserved_exclude:
                keep = torch.tensor(
                    [int(idx) not in reserved_exclude for idx in query_candidates.tolist()],
                    dtype=torch.bool,
                )
                query_candidates = query_candidates[keep]
            query_coverage_metadata_out["available_count"] = int(query_candidates.numel())
            query_budget = min(
                int(round(float(output_budget) * query_coverage_fraction)),
                int(output_budget) - int(strict_selected.numel()) - int(support_guard_selected.numel()),
                int(query_candidates.numel()),
            )
            if query_budget > 0:
                query_coverage_selected = query_candidates[:query_budget].long()
    protected_fraction = min(max(float(protected_source_fraction), 0.0), 1.0)
    protected_selected = torch.empty(0, dtype=torch.long)
    protected_metadata = dict(protected_source_metadata or {})
    if protected_source_idx is not None:
        protected_candidates, protected_counts = _ordered_unique_candidate_ids(
            protected_source_idx,
            candidate_ids=candidate_ids,
            num_gaussians=int(selector_tensor.shape[0]),
        )
        protected_metadata.update(protected_counts)
        if protected_fraction > 0.0:
            reserved_exclude = set(
                int(item)
                for item in torch.cat([strict_selected, support_guard_selected, query_coverage_selected]).tolist()
            )
            if reserved_exclude:
                keep = torch.tensor(
                    [int(idx) not in reserved_exclude for idx in protected_candidates.tolist()],
                    dtype=torch.bool,
                )
                protected_candidates = protected_candidates[keep]
            protected_metadata["available_count"] = int(protected_candidates.numel())
            protected_budget = min(
                int(round(float(output_budget) * protected_fraction)),
                int(output_budget)
                - int(strict_selected.numel())
                - int(support_guard_selected.numel())
                - int(query_coverage_selected.numel()),
                int(protected_candidates.numel()),
            )
            if protected_budget > 0:
                protected_selected = protected_candidates[:protected_budget].long()
    retention_fraction = min(max(float(source_retention_fraction), 0.0), 1.0)
    retain_count = 0
    retained = torch.empty(0, dtype=torch.long)
    if candidate_pool == "all_gaussians" and retention_fraction > 0.0:
        valid_source = source_idx[(source_idx >= 0) & (source_idx < selector_tensor.shape[0])]
        reserved_exclude = set(
            int(item)
            for item in torch.cat(
                [strict_selected, support_guard_selected, query_coverage_selected, protected_selected]
            ).tolist()
        )
        retain_count = min(
            int(round(float(output_budget) * retention_fraction)),
            int(output_budget)
            - int(strict_selected.numel())
            - int(support_guard_selected.numel())
            - int(query_coverage_selected.numel())
            - int(protected_selected.numel()),
            int(valid_source.numel()),
        )
        retained = _topk_from_candidates(
            candidate_ids=valid_source,
            scores=scores,
            xyz=xyz_tensor,
            budget=retain_count,
            coverage_grid=int(coverage_grid),
            exclude=reserved_exclude,
        )
    selected_prefix = torch.cat(
        [strict_selected, support_guard_selected, query_coverage_selected, protected_selected, retained],
        dim=0,
    )
    exclude_ids = set(int(item) for item in selected_prefix.tolist())
    fill_count = int(output_budget) - int(selected_prefix.numel())
    fill = _topk_from_candidates(
        candidate_ids=candidate_ids,
        scores=scores,
        xyz=xyz_tensor,
        budget=fill_count,
        coverage_grid=int(coverage_grid),
        exclude=exclude_ids,
    )
    selected = torch.cat([selected_prefix, fill], dim=0)
    if bool(preserve_source_order):
        source_rank = {int(gid): rank for rank, gid in enumerate(source_idx.tolist())}
        selected = torch.tensor(
            sorted(
                selected.tolist(),
                key=lambda gid: (source_rank.get(int(gid), len(source_rank)), -float(scores[int(gid)].item()), int(gid)),
            ),
            dtype=torch.long,
        )
    sampled_scores = scores[selected].float().cpu()
    result = {
        "sampled_idx": selected,
        "sampled_scores": sampled_scores,
        "score_avg": scores.float().cpu(),
        "selector": selector_tensor.float().cpu(),
        "source_count": int(source_idx.numel()),
        "output_count": int(selected.numel()),
        "sampled_idx_changed": source_idx.shape != selected.shape or not torch.equal(source_idx, selected),
        "sampling_scores": scores.float().cpu(),
        "candidate_pool": candidate_pool,
        "preserve_source_order": bool(preserve_source_order),
        "selector_transform": str(selector_transform),
        "source_retention_fraction": retention_fraction,
        "source_retained_count": int(retained.numel()),
        "strict_support_fraction": strict_fraction,
        "strict_support_reserved_count": int(strict_selected.numel()),
        "support_guard_fraction": support_guard_fraction if support_guard_idx is not None else 0.0,
        "support_guard_reserved_count": int(support_guard_selected.numel()),
        "support_guard_metadata": support_guard_metadata_out,
        "query_coverage_fraction": query_coverage_fraction if query_coverage_idx is not None else 0.0,
        "query_coverage_reserved_count": int(query_coverage_selected.numel()),
        "query_coverage_metadata": query_coverage_metadata_out,
        "protected_source_fraction": protected_fraction if protected_source_idx is not None else 0.0,
        "protected_source_reserved_count": int(protected_selected.numel()),
        "protected_source_metadata": protected_metadata,
        "hard_negative_weight": float(hard_negative_weight),
        "positive_support_weight": float(positive_support_weight),
        "pose_information_weight": float(pose_information_weight),
        "hard_query_support_weight": float(hard_query_support_weight),
    }
    if hard_negative_risk is not None:
        result["hard_negative_risk"] = _as_float_vector(hard_negative_risk, name="hard_negative_risk")
        result["hard_negative_metadata"] = dict(hard_negative_metadata or {})
    if positive_support is not None:
        result["positive_support"] = _as_float_vector(positive_support, name="positive_support")
        result["positive_support_metadata"] = dict(positive_support_metadata or {})
    if pose_information is not None:
        result["pose_information"] = _as_float_vector(pose_information, name="pose_information")
        result["pose_information_metadata"] = dict(pose_information_metadata or {})
    if hard_query_support is not None:
        result["hard_query_support"] = _as_float_vector(hard_query_support, name="hard_query_support")
        result["hard_query_support_metadata"] = dict(hard_query_support_metadata or {})
    if strict_tensor is not None:
        result["strict_support"] = strict_tensor
        result["strict_support_metadata"] = dict(strict_support_metadata or {})
    return result


def write_resampled_detector_payload(output_detector_dir: str | Path, payload: dict[str, Any]) -> None:
    output_detector_dir = Path(output_detector_dir)
    output_detector_dir.mkdir(parents=True, exist_ok=True)
    sampled_idx_path = output_detector_dir / "sampled_idx.pkl"
    sampled_scores_path = output_detector_dir / "sampled_scores.pkl"
    if sampled_idx_path.is_symlink():
        sampled_idx_path.unlink()
    if sampled_scores_path.is_symlink():
        sampled_scores_path.unlink()
    _dump_pickle(torch.as_tensor(payload["sampled_idx"], dtype=torch.long).cpu(), sampled_idx_path)
    _dump_pickle(
        {
            "sampled_scores": torch.as_tensor(payload["sampled_scores"], dtype=torch.float32).cpu(),
            "score_avg": torch.as_tensor(payload["score_avg"], dtype=torch.float32).cpu(),
            "selector": torch.as_tensor(payload.get("selector", payload["score_avg"]), dtype=torch.float32).cpu(),
            **(
                {
                    "hard_negative_risk": torch.as_tensor(
                        payload["hard_negative_risk"],
                        dtype=torch.float32,
                    ).cpu()
                }
                if "hard_negative_risk" in payload
                else {}
            ),
            **(
                {
                    "positive_support": torch.as_tensor(
                        payload["positive_support"],
                        dtype=torch.float32,
                    ).cpu()
                }
                if "positive_support" in payload
                else {}
            ),
            **(
                {
                    "pose_information": torch.as_tensor(
                        payload["pose_information"],
                        dtype=torch.float32,
                    ).cpu()
                }
                if "pose_information" in payload
                else {}
            ),
            **(
                {
                    "hard_query_support": torch.as_tensor(
                        payload["hard_query_support"],
                        dtype=torch.float32,
                    ).cpu()
                }
                if "hard_query_support" in payload
                else {}
            ),
            **(
                {
                    "strict_support": torch.as_tensor(
                        payload["strict_support"],
                        dtype=torch.float32,
                    ).cpu()
                }
                if "strict_support" in payload
                else {}
            ),
        },
        sampled_scores_path,
    )


def selector_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    num_gaussians: int,
    default: float = 0.5,
) -> torch.Tensor:
    gate = torch.as_tensor(checkpoint.get("gate", torch.empty(0)), dtype=torch.float32).reshape(-1).cpu()
    if gate.numel() == 0:
        raise KeyError("checkpoint is missing gate selector")
    ids = torch.as_tensor(checkpoint.get("base_gaussian_id", torch.empty(0)), dtype=torch.long).reshape(-1).cpu()
    if ids.numel() == 0:
        if gate.numel() != int(num_gaussians):
            raise ValueError("checkpoint gate does not cover the full source map and has no base_gaussian_id")
        ids = torch.arange(int(num_gaussians), dtype=torch.long)
    if ids.numel() != gate.numel():
        raise ValueError("base_gaussian_id must have one id per gate value")
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= int(num_gaussians)):
        raise IndexError("base_gaussian_id contains ids outside the source map")
    selector = torch.full((int(num_gaussians),), float(default), dtype=torch.float32)
    selector[ids] = gate.clamp(0.0, 1.0)
    return selector
