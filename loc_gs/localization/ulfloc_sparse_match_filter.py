from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SparseMatchFilterConfig:
    enabled: bool = False
    mode: str = "precision"
    min_query_margin: float = 0.0
    min_descriptor_score: float = -float("inf")
    top_m: int = 0
    min_keep: int = 0
    unique_landmark: bool = False
    margin_weight: float = 1.0
    detector_score_weight: float = 0.0
    landmark_prior_weights: torch.Tensor | None = None
    landmark_prior_weight: float = 0.0
    landmark_prior_center: float = 0.0
    landmark_prior_scale: float = 1.0
    landmark_prior_clip: float = 0.0
    image_grid_size: int = 0
    max_per_image_cell: int = 0

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"none", "precision", "margin", "descriptor_margin", "reorder"}:
            raise ValueError(f"unsupported sparse match filter mode: {self.mode}")


@dataclass(frozen=True)
class SparseMatchFilterResult:
    im_idx: torch.Tensor
    gs_ids: torch.Tensor
    descriptor_scores: torch.Tensor
    keep_positions: torch.Tensor
    query_margins: torch.Tensor
    combined_scores: torch.Tensor
    metadata: dict[str, Any]


def _as_vector(values: torch.Tensor, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype).reshape(-1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return tensor


def _selected_detector_scores(
    detector_scores: torch.Tensor | None,
    im_idx: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if detector_scores is None:
        return torch.zeros((int(im_idx.numel()),), device=device, dtype=dtype)
    scores = torch.as_tensor(detector_scores, dtype=dtype, device=device).reshape(-1)
    if int(scores.numel()) == int(im_idx.numel()):
        return scores
    if int(im_idx.numel()) == 0:
        return torch.empty((0,), device=device, dtype=dtype)
    if int(scores.numel()) <= int(im_idx.max().item()):
        return torch.zeros((int(im_idx.numel()),), device=device, dtype=dtype)
    return scores[im_idx]


def _query_second_best_margins(
    corr_matrix: torch.Tensor,
    im_idx: torch.Tensor,
    gs_ids: torch.Tensor,
    descriptor_scores: torch.Tensor,
) -> torch.Tensor:
    if int(im_idx.numel()) == 0:
        return descriptor_scores.new_empty((0,))
    if corr_matrix.ndim != 2:
        raise ValueError("corr_matrix must have shape [query_count, landmark_count]")
    if int(corr_matrix.shape[1]) <= 1:
        return torch.full_like(descriptor_scores, float("inf"))
    if int(im_idx.min().item()) < 0 or int(im_idx.max().item()) >= int(corr_matrix.shape[0]):
        raise ValueError("im_idx contains out-of-range query indices")
    if int(gs_ids.min().item()) < 0 or int(gs_ids.max().item()) >= int(corr_matrix.shape[1]):
        raise ValueError("gs_ids contains out-of-range landmark indices")

    rows = corr_matrix[im_idx].to(dtype=descriptor_scores.dtype)
    masked = rows.clone()
    masked[torch.arange(int(gs_ids.numel()), device=gs_ids.device), gs_ids] = -float("inf")
    second_best = masked.max(dim=1).values
    return descriptor_scores - second_best


def _keep_one_per_landmark(positions: torch.Tensor, gs_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if int(positions.numel()) <= 1:
        return positions
    best: dict[int, tuple[float, int]] = {}
    for pos in positions.detach().cpu().tolist():
        gid = int(gs_ids[pos].item())
        score = float(scores[pos].item())
        previous = best.get(gid)
        if previous is None or score > previous[0] or (score == previous[0] and int(pos) < previous[1]):
            best[gid] = (score, int(pos))
    kept = sorted(item[1] for item in best.values())
    return torch.as_tensor(kept, dtype=torch.long, device=positions.device)


def _top_positions(positions: torch.Tensor, scores: torch.Tensor, *, limit: int) -> torch.Tensor:
    if int(limit) <= 0 or int(positions.numel()) <= int(limit):
        return positions
    order = torch.argsort(scores[positions], descending=True)[: int(limit)]
    return positions[order]


def _selected_keypoint_xy(
    keypoint_xy: torch.Tensor | None,
    im_idx: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if keypoint_xy is None:
        return None
    xy = torch.as_tensor(keypoint_xy, dtype=torch.float32, device=device)
    if xy.ndim != 2 or int(xy.shape[1]) < 2:
        raise ValueError("keypoint_xy must have shape [N,2]")
    xy = xy[:, :2]
    if int(xy.shape[0]) == int(im_idx.numel()):
        return xy
    if int(im_idx.numel()) == 0:
        return xy.new_empty((0, 2))
    if int(xy.shape[0]) <= int(im_idx.max().item()):
        raise ValueError("keypoint_xy does not cover im_idx")
    return xy[im_idx]


def _selected_landmark_prior_scores(
    landmark_prior_weights: torch.Tensor | None,
    gs_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if landmark_prior_weights is None or int(gs_ids.numel()) == 0:
        return torch.zeros((int(gs_ids.numel()),), device=device, dtype=dtype)
    prior = torch.as_tensor(landmark_prior_weights, dtype=dtype, device=device).reshape(-1)
    if int(prior.numel()) == 0:
        return torch.zeros((int(gs_ids.numel()),), device=device, dtype=dtype)
    out = torch.zeros((int(gs_ids.numel()),), device=device, dtype=dtype)
    valid = (gs_ids >= 0) & (gs_ids < int(prior.numel()))
    if bool(valid.any()):
        out[valid] = prior[gs_ids[valid]]
    return out


def prepare_selected_landmark_prior_weights(
    landmark_prior_weights: torch.Tensor | None,
    *,
    sampled_idx: torch.Tensor | None,
    selected_count: int,
) -> torch.Tensor | None:
    """Map full-Gaussian prior weights into ULF's sampled-landmark column space."""
    if landmark_prior_weights is None:
        return None
    prior = torch.as_tensor(landmark_prior_weights, dtype=torch.float32).reshape(-1).cpu()
    selected_count = int(selected_count)
    if selected_count <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if int(prior.numel()) == selected_count:
        return prior.clone()
    if sampled_idx is None:
        if int(prior.numel()) < selected_count:
            return None
        return prior[:selected_count].clone()
    sampled = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    if int(sampled.numel()) != selected_count or int(sampled.numel()) == 0:
        return None
    valid = (sampled >= 0) & (sampled < int(prior.numel()))
    selected = torch.zeros((selected_count,), dtype=torch.float32)
    if bool(valid.any()):
        selected[valid] = prior[sampled[valid]]
    return selected


def calibrate_landmark_prior_scores(
    prior_scores: torch.Tensor,
    *,
    center: float = 0.0,
    scale: float = 1.0,
    clip: float = 0.0,
) -> torch.Tensor:
    scores = torch.as_tensor(prior_scores, dtype=torch.float32)
    denom = max(float(scale), 1e-6)
    calibrated = (scores - float(center)) / denom
    if float(clip) > 0.0:
        calibrated = calibrated.clamp(min=-float(clip), max=float(clip))
    return calibrated


def adjust_corr_matrix_with_landmark_prior(
    corr_matrix: torch.Tensor,
    landmark_prior_weights: torch.Tensor | None,
    *,
    weight: float,
    center: float = 0.0,
    scale: float = 1.0,
    clip: float = 0.0,
) -> torch.Tensor:
    if landmark_prior_weights is None or float(weight) == 0.0:
        return corr_matrix
    corr = torch.as_tensor(corr_matrix)
    prior = torch.as_tensor(landmark_prior_weights, dtype=corr.dtype, device=corr.device).reshape(-1)
    if int(prior.numel()) < int(corr.shape[-1]):
        padded = torch.zeros((int(corr.shape[-1]),), dtype=corr.dtype, device=corr.device)
        if int(prior.numel()) > 0:
            padded[: int(prior.numel())] = prior
        prior = padded
    else:
        prior = prior[: int(corr.shape[-1])]
    calibrated = calibrate_landmark_prior_scores(
        prior,
        center=float(center),
        scale=float(scale),
        clip=float(clip),
    ).to(device=corr.device, dtype=corr.dtype)
    return corr + float(weight) * calibrated.reshape((1,) * (corr.ndim - 1) + (-1,))


def _query_topk_margin(top_values: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    values = top_values.float().masked_fill(~candidate_mask, float("-inf"))
    if int(values.shape[1]) <= 1:
        return torch.zeros((int(values.shape[0]),), dtype=values.dtype, device=values.device)
    sorted_values = torch.sort(values, dim=1, descending=True).values
    best = sorted_values[:, 0]
    second = sorted_values[:, 1]
    margin = best - second
    return torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin))


def select_scene_matcher_topk_matches(
    *,
    corr_matrix: torch.Tensor,
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    detector_scores: torch.Tensor | None,
    matcher: torch.nn.Module,
    topk: int = 4,
    score_weight: float = 0.0,
    threshold: float = -float("inf"),
    landmark_prior_weights: torch.Tensor | None = None,
    landmark_prior_center: float = 0.0,
    landmark_prior_scale: float = 1.0,
    landmark_prior_clip: float = 0.0,
    drop_dustbin: bool = False,
    scene_score_mode: str = "candidate_minus_dustbin",
    max_descriptor_margin: float = float("inf"),
    min_keep: int = 0,
) -> SparseMatchFilterResult:
    """Select sparse correspondences with a solver-trained top-K listwise matcher.

    The matcher only changes the 2D-keypoint-to-3D-landmark correspondence before
    the normal PnP path. It does not choose among pose hypotheses or use GT at
    inference.
    """
    from loc_gs.localization.scene_matcher import score_scene_match_candidates

    score_mode = str(scene_score_mode).strip().lower()
    if score_mode not in {"candidate_minus_dustbin", "candidate"}:
        raise ValueError(f"unsupported scene_score_mode: {scene_score_mode}")
    corr = torch.as_tensor(corr_matrix, dtype=torch.float32)
    if corr.ndim != 2:
        raise ValueError("corr_matrix must have shape [query_count, landmark_count]")
    q_desc = torch.as_tensor(query_desc, dtype=torch.float32, device=corr.device)
    lm_desc = torch.as_tensor(landmark_desc, dtype=torch.float32, device=corr.device)
    if q_desc.ndim != 2 or lm_desc.ndim != 2:
        raise ValueError("query_desc and landmark_desc must be [N,D] and [M,D]")
    if int(q_desc.shape[0]) != int(corr.shape[0]) or int(lm_desc.shape[0]) != int(corr.shape[1]):
        raise ValueError("descriptor rows must match corr_matrix dimensions")
    if str(getattr(matcher, "config", {}).get("model_type", "pairwise")) != "listwise":
        raise ValueError("scene matcher sparse top-k selection requires a listwise matcher")

    query_count, landmark_count = int(corr.shape[0]), int(corr.shape[1])
    if query_count == 0 or landmark_count == 0:
        empty_long = torch.empty((0,), dtype=torch.long, device=corr.device)
        empty_float = torch.empty((0,), dtype=corr.dtype, device=corr.device)
        return SparseMatchFilterResult(
            im_idx=empty_long,
            gs_ids=empty_long,
            descriptor_scores=empty_float,
            keep_positions=empty_long,
            query_margins=empty_float,
            combined_scores=empty_float,
            metadata={
                "method": "scene_matcher_topk_correspondence_selection",
                "scene_matcher_enabled": True,
                "scene_matcher_topk": int(topk),
                "scene_matcher_score_mode": score_mode,
                "scene_matcher_max_descriptor_margin": float(max_descriptor_margin),
                "scene_matcher_margin_guard_count": 0,
                "scene_matcher_min_keep": int(min_keep),
                "scene_matcher_min_keep_backfill_count": 0,
                "scene_matcher_selected_count": 0,
            },
        )

    k = min(max(1, int(topk)), landmark_count)
    top_values, top_ids = torch.topk(corr, k=k, dim=1)
    candidate_mask = top_values > float(threshold)
    margins = _query_topk_margin(top_values, candidate_mask)
    if landmark_prior_weights is None:
        prior = torch.zeros((query_count, k), dtype=corr.dtype, device=corr.device)
        calibrated_prior = prior
    else:
        full_prior = torch.as_tensor(landmark_prior_weights, dtype=corr.dtype, device=corr.device).reshape(-1)
        safe_ids = top_ids.clamp(min=0, max=max(0, int(full_prior.numel()) - 1))
        prior = full_prior[safe_ids] if int(full_prior.numel()) > 0 else torch.zeros_like(top_values)
        calibrated_prior = calibrate_landmark_prior_scores(
            prior,
            center=float(landmark_prior_center),
            scale=float(landmark_prior_scale),
            clip=float(landmark_prior_clip),
        ).to(device=corr.device, dtype=corr.dtype)

    query_score = None
    if detector_scores is not None:
        detector = torch.as_tensor(detector_scores, dtype=corr.dtype, device=corr.device).reshape(-1)
        if int(detector.numel()) == query_count:
            query_score = detector

    logits = score_scene_match_candidates(
        matcher.to(corr.device),
        q_desc,
        lm_desc[top_ids],
        cosine=top_values,
        margin=margins,
        landmark_prior=prior,
        calibrated_prior=calibrated_prior,
        query_score=query_score,
        candidate_mask=candidate_mask,
    )
    candidate_logits = logits[:, :k]
    dustbin_logits = logits[:, k]
    if score_mode == "candidate":
        scene_delta = candidate_logits
    else:
        scene_delta = candidate_logits - dustbin_logits[:, None]
    margin_guard = margins > float(max_descriptor_margin)
    if bool(margin_guard.any()):
        scene_delta = scene_delta.masked_fill(margin_guard[:, None], 0.0)
    combined = top_values + float(score_weight) * scene_delta
    combined = combined.masked_fill(~candidate_mask, float("-inf"))
    combined_no_drop = combined.clone()
    if bool(drop_dustbin):
        combined = combined.masked_fill(candidate_logits <= dustbin_logits[:, None], float("-inf"))
    best_scores, best_cols = combined.max(dim=1)
    valid = torch.isfinite(best_scores)
    no_drop_best_scores, no_drop_best_cols = combined_no_drop.max(dim=1)
    no_drop_valid = torch.isfinite(no_drop_best_scores)

    raw_drop_count = int(query_count - int(valid.sum().item())) if bool(drop_dustbin) else 0
    min_keep_count = min(query_count, max(0, int(min_keep)))
    min_keep_backfill = 0
    if int(valid.sum().item()) < min_keep_count and bool(no_drop_valid.any()):
        selected_rows = set(torch.arange(query_count, device=corr.device)[valid].detach().cpu().tolist())
        available = [
            row
            for row in range(query_count)
            if row not in selected_rows and bool(no_drop_valid[row].item())
        ]
        if available:
            available_rows = torch.as_tensor(available, dtype=torch.long, device=corr.device)
            order = torch.argsort(no_drop_best_scores[available_rows], descending=True)
            need = min_keep_count - int(valid.sum().item())
            backfill_rows = available_rows[order[:need]]
            valid[backfill_rows] = True
            best_scores[backfill_rows] = no_drop_best_scores[backfill_rows]
            best_cols[backfill_rows] = no_drop_best_cols[backfill_rows]
            min_keep_backfill = int(backfill_rows.numel())

    if not bool(valid.any()):
        empty_long = torch.empty((0,), dtype=torch.long, device=corr.device)
        empty_float = torch.empty((0,), dtype=corr.dtype, device=corr.device)
        return SparseMatchFilterResult(
            im_idx=empty_long,
            gs_ids=empty_long,
            descriptor_scores=empty_float,
            keep_positions=empty_long,
            query_margins=empty_float,
            combined_scores=empty_float,
            metadata={
                "method": "scene_matcher_topk_correspondence_selection",
                "scene_matcher_enabled": True,
                "scene_matcher_topk": int(k),
                "scene_matcher_score_mode": score_mode,
                "scene_matcher_max_descriptor_margin": float(max_descriptor_margin),
                "scene_matcher_margin_guard_count": 0,
                "scene_matcher_min_keep": int(min_keep),
                "scene_matcher_min_keep_backfill_count": 0,
                "scene_matcher_selected_count": 0,
                "scene_matcher_drop_dustbin": bool(drop_dustbin),
            },
        )
    rows = torch.arange(query_count, dtype=torch.long, device=corr.device)[valid]
    cols = best_cols[valid]
    gs = top_ids[rows, cols]
    descriptor_scores = top_values[rows, cols]
    combined_scores = best_scores[valid]
    order = torch.argsort(combined_scores, descending=True)
    rows = rows[order]
    gs = gs[order]
    descriptor_scores = descriptor_scores[order]
    combined_scores = combined_scores[order]
    selected_margins = margins[rows]
    keep_positions = rows * k + cols[order]
    return SparseMatchFilterResult(
        im_idx=rows,
        gs_ids=gs,
        descriptor_scores=descriptor_scores,
        keep_positions=keep_positions,
        query_margins=selected_margins,
        combined_scores=combined_scores,
        metadata={
            "method": "scene_matcher_topk_correspondence_selection",
            "scene_matcher_enabled": True,
            "scene_matcher_topk": int(k),
            "scene_matcher_score_weight": float(score_weight),
            "scene_matcher_score_mode": score_mode,
            "scene_matcher_drop_dustbin": bool(drop_dustbin),
            "scene_matcher_max_descriptor_margin": float(max_descriptor_margin),
            "scene_matcher_margin_guard_count": int(margin_guard.sum().item()),
            "scene_matcher_min_keep": int(min_keep),
            "scene_matcher_min_keep_backfill_count": int(min_keep_backfill),
            "scene_matcher_selected_count": int(rows.numel()),
            "scene_matcher_input_query_count": int(query_count),
            "scene_matcher_candidate_count": int(candidate_mask.sum().item()),
            "scene_matcher_dustbin_raw_drop_count": int(raw_drop_count),
            "scene_matcher_dustbin_drop_count": int(query_count - rows.numel()),
            "scene_matcher_mean_delta": float(scene_delta[candidate_mask].mean().item())
            if bool(candidate_mask.any())
            else 0.0,
            "input_match_count": int(query_count * k),
            "output_match_count": int(rows.numel()),
        },
    )


def _cap_positions_per_image_cell(
    positions: torch.Tensor,
    scores: torch.Tensor,
    *,
    selected_xy: torch.Tensor | None,
    image_size: tuple[int, int] | None,
    grid_size: int,
    max_per_cell: int,
) -> torch.Tensor:
    if (
        selected_xy is None
        or image_size is None
        or int(grid_size) <= 0
        or int(max_per_cell) <= 0
        or int(positions.numel()) <= int(max_per_cell)
    ):
        return positions
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive [width,height]")
    grid = int(grid_size)
    xy = selected_xy.to(device=positions.device, dtype=torch.float32)
    x_cell = torch.clamp((xy[:, 0] / float(width) * grid).floor().to(torch.long), 0, grid - 1)
    y_cell = torch.clamp((xy[:, 1] / float(height) * grid).floor().to(torch.long), 0, grid - 1)
    cell_ids = y_cell * grid + x_cell
    kept: list[int] = []
    for raw_cell in torch.unique(cell_ids[positions]).detach().cpu().tolist():
        cell = int(raw_cell)
        cell_positions = positions[cell_ids[positions] == cell]
        top = _top_positions(cell_positions, scores, limit=int(max_per_cell))
        kept.extend(int(pos) for pos in top.detach().cpu().tolist())
    return torch.as_tensor(sorted(kept), dtype=torch.long, device=positions.device)


def filter_ulfloc_sparse_matches(
    *,
    corr_matrix: torch.Tensor,
    im_idx: torch.Tensor,
    gs_ids: torch.Tensor,
    descriptor_scores: torch.Tensor,
    detector_scores: torch.Tensor | None = None,
    keypoint_xy: torch.Tensor | None = None,
    image_size: tuple[int, int] | None = None,
    config: SparseMatchFilterConfig | None = None,
) -> SparseMatchFilterResult:
    cfg = config or SparseMatchFilterConfig(enabled=False)
    im = _as_vector(im_idx, dtype=torch.long, name="im_idx")
    gs = _as_vector(gs_ids, dtype=torch.long, name="gs_ids").to(device=im.device)
    scores = _as_vector(descriptor_scores, dtype=torch.float32, name="descriptor_scores").to(device=im.device)
    if int(im.numel()) != int(gs.numel()) or int(im.numel()) != int(scores.numel()):
        raise ValueError("im_idx, gs_ids, and descriptor_scores must have the same length")

    input_count = int(im.numel())
    if input_count == 0 or not bool(cfg.enabled) or str(cfg.mode).strip().lower() == "none":
        keep_positions = torch.arange(input_count, dtype=torch.long, device=im.device)
        margins = torch.zeros_like(scores)
        return SparseMatchFilterResult(
            im_idx=im,
            gs_ids=gs,
            descriptor_scores=scores,
            keep_positions=keep_positions,
            query_margins=margins,
            combined_scores=scores,
            metadata={
                "method": "ulfloc_sparse_match_precision_filter",
                "enabled": bool(cfg.enabled),
                "mode": str(cfg.mode),
                "input_match_count": input_count,
                "output_match_count": input_count,
                "removed_low_margin_count": 0,
                "removed_low_score_count": 0,
                "unique_landmark_removed_count": 0,
                "top_m_removed_count": 0,
                "image_cell_cap_removed_count": 0,
                "min_keep_backfill_count": 0,
            },
        )

    corr = torch.as_tensor(corr_matrix, dtype=torch.float32, device=im.device)
    margins = _query_second_best_margins(corr, im, gs, scores)
    selected_detector = _selected_detector_scores(
        detector_scores,
        im,
        device=im.device,
        dtype=scores.dtype,
    )
    selected_landmark_prior = _selected_landmark_prior_scores(
        cfg.landmark_prior_weights,
        gs,
        device=im.device,
        dtype=scores.dtype,
    )
    selected_landmark_prior = calibrate_landmark_prior_scores(
        selected_landmark_prior,
        center=float(cfg.landmark_prior_center),
        scale=float(cfg.landmark_prior_scale),
        clip=float(cfg.landmark_prior_clip),
    ).to(device=im.device, dtype=scores.dtype)
    combined = (
        scores
        + float(cfg.margin_weight) * margins.clamp_min(0.0)
        + float(cfg.detector_score_weight) * selected_detector
        + float(cfg.landmark_prior_weight) * selected_landmark_prior
    )

    if str(cfg.mode).strip().lower() == "reorder":
        positions = torch.argsort(combined, descending=True)
        return SparseMatchFilterResult(
            im_idx=im[positions],
            gs_ids=gs[positions],
            descriptor_scores=scores[positions],
            keep_positions=positions,
            query_margins=margins,
            combined_scores=combined,
            metadata={
                "method": "ulfloc_sparse_match_precision_filter",
                "enabled": bool(cfg.enabled),
                "mode": str(cfg.mode),
                "input_match_count": input_count,
                "output_match_count": input_count,
                "removed_low_margin_count": 0,
                "removed_low_score_count": 0,
                "unique_landmark_removed_count": 0,
                "top_m_removed_count": 0,
                "image_cell_cap_removed_count": 0,
                "min_keep_backfill_count": 0,
                "reordered_for_prosac": True,
                "margin_weight": float(cfg.margin_weight),
                "detector_score_weight": float(cfg.detector_score_weight),
                "landmark_prior_weight": float(cfg.landmark_prior_weight),
                "landmark_prior_center": float(cfg.landmark_prior_center),
                "landmark_prior_scale": float(cfg.landmark_prior_scale),
                "landmark_prior_clip": float(cfg.landmark_prior_clip),
                "mean_landmark_prior_score": float(selected_landmark_prior.mean().item())
                if int(selected_landmark_prior.numel())
                else 0.0,
                "mean_query_margin": float(margins.mean().item()) if int(margins.numel()) else 0.0,
                "mean_combined_score": float(combined.mean().item()) if int(combined.numel()) else 0.0,
            },
        )

    low_margin = margins < float(cfg.min_query_margin)
    low_score = scores < float(cfg.min_descriptor_score)
    keep_mask = ~(low_margin | low_score)
    positions = torch.where(keep_mask)[0]

    unique_removed = 0
    if bool(cfg.unique_landmark) and int(positions.numel()) > 0:
        before_unique = int(positions.numel())
        positions = _keep_one_per_landmark(positions, gs, combined)
        unique_removed = before_unique - int(positions.numel())

    top_m_removed = 0
    if int(cfg.top_m) > 0 and int(positions.numel()) > int(cfg.top_m):
        before_top = int(positions.numel())
        positions = _top_positions(positions, combined, limit=int(cfg.top_m))
        top_m_removed = before_top - int(positions.numel())

    image_cell_cap_removed = 0
    selected_xy = _selected_keypoint_xy(keypoint_xy, im, device=im.device)
    if int(cfg.image_grid_size) > 0 and int(cfg.max_per_image_cell) > 0 and int(positions.numel()) > 0:
        before_cell = int(positions.numel())
        positions = _cap_positions_per_image_cell(
            positions,
            combined,
            selected_xy=selected_xy,
            image_size=image_size,
            grid_size=int(cfg.image_grid_size),
            max_per_cell=int(cfg.max_per_image_cell),
        )
        image_cell_cap_removed = before_cell - int(positions.numel())

    min_keep_backfill = 0
    min_keep = min(input_count, max(0, int(cfg.min_keep)))
    if int(positions.numel()) < min_keep:
        already = torch.zeros((input_count,), dtype=torch.bool, device=im.device)
        if int(positions.numel()) > 0:
            already[positions] = True
        candidate_positions = torch.where(~already)[0]
        need = min_keep - int(positions.numel())
        if need > 0 and int(candidate_positions.numel()) > 0:
            backfill = _top_positions(candidate_positions, combined, limit=need)
            positions = torch.cat([positions, backfill], dim=0)
            min_keep_backfill = int(backfill.numel())

    positions = positions.sort().values
    return SparseMatchFilterResult(
        im_idx=im[positions],
        gs_ids=gs[positions],
        descriptor_scores=scores[positions],
        keep_positions=positions,
        query_margins=margins,
        combined_scores=combined,
        metadata={
            "method": "ulfloc_sparse_match_precision_filter",
            "enabled": bool(cfg.enabled),
            "mode": str(cfg.mode),
            "input_match_count": input_count,
            "output_match_count": int(positions.numel()),
            "removed_low_margin_count": int(low_margin.sum().item()),
            "removed_low_score_count": int(low_score.sum().item()),
            "unique_landmark_removed_count": int(unique_removed),
            "top_m_removed_count": int(top_m_removed),
            "image_cell_cap_removed_count": int(image_cell_cap_removed),
            "min_keep_backfill_count": int(min_keep_backfill),
            "reordered_for_prosac": False,
            "min_query_margin": float(cfg.min_query_margin),
            "min_descriptor_score": float(cfg.min_descriptor_score),
            "top_m": int(cfg.top_m),
            "min_keep": int(cfg.min_keep),
            "unique_landmark": bool(cfg.unique_landmark),
            "image_grid_size": int(cfg.image_grid_size),
            "max_per_image_cell": int(cfg.max_per_image_cell),
            "landmark_prior_weight": float(cfg.landmark_prior_weight),
            "landmark_prior_center": float(cfg.landmark_prior_center),
            "landmark_prior_scale": float(cfg.landmark_prior_scale),
            "landmark_prior_clip": float(cfg.landmark_prior_clip),
            "mean_landmark_prior_score": float(selected_landmark_prior.mean().item())
            if int(selected_landmark_prior.numel())
            else 0.0,
            "mean_query_margin": float(margins.mean().item()) if int(margins.numel()) else 0.0,
            "mean_combined_score": float(combined.mean().item()) if int(combined.numel()) else 0.0,
        },
    )
