from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def dual_softmax(corr_matrix: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """STDLoc-style dual-softmax normalization over a correlation matrix."""
    scaled = corr_matrix / max(float(temp), 1e-6)
    return F.softmax(scaled, dim=-2) * F.softmax(scaled, dim=-1)


def apply_match_prior(
    corr_matrix: torch.Tensor,
    prior: Optional[torch.Tensor],
    weight: float = 0.0,
) -> torch.Tensor:
    """Add a centered landmark/rendered locability prior to match logits."""
    if prior is None or float(weight) == 0.0:
        return corr_matrix
    prior_flat = prior.to(device=corr_matrix.device, dtype=corr_matrix.dtype).reshape(-1)
    if prior_flat.shape[0] != corr_matrix.shape[-1]:
        return corr_matrix
    prior_flat = prior_flat.clamp(0.0, 1.0)
    prior_flat = prior_flat - prior_flat.mean()
    return corr_matrix + float(weight) * prior_flat.reshape(*([1] * (corr_matrix.dim() - 1)), -1)


def mnn_match(
    corr_matrix: torch.Tensor,
    threshold: float = -1.0,
    second_best_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mutual-nearest-neighbor matches from [B, N, M] scores."""
    if corr_matrix.dim() != 3:
        raise ValueError("corr_matrix must have shape [B, N, M]")
    row_best = corr_matrix == corr_matrix.max(dim=-1, keepdim=True).values
    col_best = corr_matrix == corr_matrix.max(dim=-2, keepdim=True).values
    keep = row_best & col_best & (corr_matrix > float(threshold))
    margin = max(float(second_best_margin), 0.0)
    if margin > 0.0 and corr_matrix.shape[-1] > 1:
        top2 = torch.topk(corr_matrix, k=2, dim=-1).values
        row_margin = top2[..., 0] - top2[..., 1]
        keep = keep & (row_margin[..., None] >= margin)
    b_ids, q_ids, r_ids = torch.where(keep)
    scores = corr_matrix[b_ids, q_ids, r_ids]
    return b_ids, q_ids, r_ids, scores


def topk_match(
    corr_matrix: torch.Tensor,
    topk: int = 1,
    threshold: float = -1.0,
    second_best_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-query top-k matches from [B, N, M] scores."""
    if corr_matrix.dim() != 3:
        raise ValueError("corr_matrix must have shape [B, N, M]")
    if corr_matrix.shape[-1] == 0 or corr_matrix.shape[-2] == 0:
        device = corr_matrix.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, corr_matrix.new_empty(0)
    k = min(max(1, int(topk)), corr_matrix.shape[-1])
    vals, ids = torch.topk(corr_matrix, k=k, dim=-1)
    B, N, _ = vals.shape
    b_grid = torch.arange(B, device=corr_matrix.device).view(B, 1, 1).expand(B, N, k)
    q_grid = torch.arange(N, device=corr_matrix.device).view(1, N, 1).expand(B, N, k)
    keep = vals > float(threshold)
    margin = max(float(second_best_margin), 0.0)
    if margin > 0.0 and corr_matrix.shape[-1] > k:
        vals_all = torch.topk(corr_matrix, k=k + 1, dim=-1).values
        next_best = vals_all[..., k].unsqueeze(-1).expand_as(vals)
        keep = keep & ((vals - next_best) >= margin)
    return b_grid[keep], q_grid[keep], ids[keep], vals[keep]


def match_correlation_matrix(
    corr_matrix: torch.Tensor,
    threshold: float = -1.0,
    dual_softmax_temp: float = 0.1,
    use_dual_softmax: bool = False,
    use_mnn: bool = False,
    topk: int = 1,
    second_best_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = (
        dual_softmax(corr_matrix, temp=dual_softmax_temp)
        if use_dual_softmax
        else corr_matrix
    )
    if use_mnn:
        return mnn_match(scores, threshold=threshold, second_best_margin=second_best_margin)
    return topk_match(scores, topk=topk, threshold=threshold, second_best_margin=second_best_margin)


def soft_argmax_offsets(
    scores: torch.Tensor,
    window_size: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Return expected [y, x] offset inside a square fine window."""
    if scores.numel() == 0:
        return scores.new_zeros((0, 2))
    W = int(window_size)
    if scores.shape[-1] != W * W:
        raise ValueError("scores last dimension must equal window_size ** 2")
    weights = F.softmax(scores / max(float(temperature), 1e-6), dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(W, device=scores.device, dtype=scores.dtype),
        torch.arange(W, device=scores.device, dtype=scores.dtype),
        indexing="ij",
    )
    coords_yx = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
    return weights @ coords_yx


@dataclass
class DenseMatchResult:
    query_yx: torch.Tensor
    rendered_yx: torch.Tensor
    scores: torch.Tensor
    coarse_query_ids: torch.Tensor
    coarse_rendered_ids: torch.Tensor


def _default_coarse(feature_map: torch.Tensor, window_size: int) -> torch.Tensor:
    pooled = F.avg_pool2d(
        feature_map.unsqueeze(0),
        kernel_size=int(window_size),
        stride=int(window_size),
    )[0]
    return F.normalize(pooled.float(), p=2, dim=0)


def coarse_to_fine_dense_matches(
    query_fine_map: torch.Tensor,
    rendered_fine_map: torch.Tensor,
    query_coarse_map: Optional[torch.Tensor] = None,
    rendered_coarse_map: Optional[torch.Tensor] = None,
    rendered_prior: Optional[torch.Tensor] = None,
    prior_weight: float = 0.0,
    window_size: int = 8,
    coarse_dual_softmax_temp: float = 0.1,
    fine_dual_softmax_temp: float = 0.1,
    coarse_threshold: float = 0.0,
    fine_threshold: float = 0.0,
    use_mnn: bool = True,
    subpixel_refine: bool = True,
    subpixel_temperature: float = 0.1,
) -> DenseMatchResult:
    """STDLoc-style coarse-to-fine dense matching for one image pair."""
    if query_fine_map.shape != rendered_fine_map.shape:
        raise ValueError("query_fine_map and rendered_fine_map must have the same shape")
    C, Hf, Wf = query_fine_map.shape
    W = max(1, int(window_size))
    crop_h = (Hf // W) * W
    crop_w = (Wf // W) * W
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("fine map is smaller than window_size")
    if crop_h != Hf or crop_w != Wf:
        query_fine_map = query_fine_map[:, :crop_h, :crop_w]
        rendered_fine_map = rendered_fine_map[:, :crop_h, :crop_w]
        if rendered_prior is not None:
            rendered_prior = rendered_prior.reshape(Hf, Wf)[:crop_h, :crop_w]
    Hf, Wf = crop_h, crop_w
    Hc, Wc = Hf // W, Wf // W

    query_fine = F.normalize(query_fine_map.float(), p=2, dim=0)
    rendered_fine = F.normalize(rendered_fine_map.float(), p=2, dim=0)
    query_coarse = (
        _default_coarse(query_fine, W)
        if query_coarse_map is None
        else F.normalize(query_coarse_map.float(), p=2, dim=0)
    )
    rendered_coarse = (
        _default_coarse(rendered_fine, W)
        if rendered_coarse_map is None
        else F.normalize(rendered_coarse_map.float(), p=2, dim=0)
    )

    coarse_corr = torch.matmul(
        query_coarse.permute(1, 2, 0).reshape(1, -1, C),
        rendered_coarse.reshape(1, C, -1),
    )
    coarse_prior = None
    if rendered_prior is not None:
        coarse_prior = F.interpolate(
            rendered_prior.reshape(1, 1, Hf, Wf).float(),
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    coarse_corr = apply_match_prior(coarse_corr, coarse_prior, weight=prior_weight)
    c_b, c_q, c_r, _c_scores = match_correlation_matrix(
        coarse_corr,
        threshold=coarse_threshold,
        dual_softmax_temp=coarse_dual_softmax_temp,
        use_dual_softmax=True,
        use_mnn=use_mnn,
        topk=1,
    )
    if c_q.numel() == 0:
        empty = query_fine.new_empty((0, 2))
        empty_l = torch.empty(0, dtype=torch.long, device=query_fine.device)
        return DenseMatchResult(empty, empty, query_fine.new_empty(0), empty_l, empty_l)

    query_windows = F.unfold(query_fine.unsqueeze(0), kernel_size=(W, W), stride=W)
    query_windows = query_windows.reshape(1, C, W * W, -1)[c_b, :, :, c_q].permute(0, 2, 1)
    rendered_windows = F.unfold(rendered_fine.unsqueeze(0), kernel_size=(W, W), stride=W)
    rendered_windows = rendered_windows.reshape(1, C, W * W, -1)[c_b, :, :, c_r].permute(0, 2, 1)
    fine_scores_raw = torch.matmul(query_windows, rendered_windows.transpose(-2, -1))
    if rendered_prior is not None and float(prior_weight) != 0.0:
        rendered_prior_windows = F.unfold(
            rendered_prior.reshape(1, 1, Hf, Wf).float(),
            kernel_size=(W, W),
            stride=W,
        )
        rendered_prior_windows = rendered_prior_windows.reshape(1, 1, W * W, -1)[c_b, :, :, c_r].squeeze(1)
        rendered_prior_windows = rendered_prior_windows.clamp(0.0, 1.0)
        rendered_prior_windows = rendered_prior_windows - rendered_prior_windows.mean(dim=-1, keepdim=True)
        fine_scores_raw = fine_scores_raw + float(prior_weight) * rendered_prior_windows[:, None, :]
    fine_corr = dual_softmax(fine_scores_raw, temp=fine_dual_softmax_temp)
    f_b, f_q, f_r, f_scores = mnn_match(fine_corr, threshold=fine_threshold)
    if f_q.numel() == 0:
        empty = query_fine.new_empty((0, 2))
        empty_l = torch.empty(0, dtype=torch.long, device=query_fine.device)
        return DenseMatchResult(empty, empty, query_fine.new_empty(0), empty_l, empty_l)

    if subpixel_refine:
        q_offsets = soft_argmax_offsets(
            fine_scores_raw[f_b, :, f_r],
            window_size=W,
            temperature=subpixel_temperature,
        )
        r_offsets = soft_argmax_offsets(
            fine_scores_raw[f_b, f_q, :],
            window_size=W,
            temperature=subpixel_temperature,
        )
    else:
        q_offsets = torch.stack([f_q // W, f_q % W], dim=-1).to(dtype=query_fine.dtype)
        r_offsets = torch.stack([f_r // W, f_r % W], dim=-1).to(dtype=query_fine.dtype)

    coarse_q = c_q[f_b]
    coarse_r = c_r[f_b]
    q_origins = torch.stack([coarse_q // Wc * W, coarse_q % Wc * W], dim=-1).to(dtype=query_fine.dtype)
    r_origins = torch.stack([coarse_r // Wc * W, coarse_r % Wc * W], dim=-1).to(dtype=query_fine.dtype)
    return DenseMatchResult(
        query_yx=q_origins + q_offsets,
        rendered_yx=r_origins + r_offsets,
        scores=f_scores,
        coarse_query_ids=coarse_q,
        coarse_rendered_ids=coarse_r,
    )
