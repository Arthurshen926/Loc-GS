from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from loc_gs.localization.selfmap_episode_cache import build_selfmap_episode_cache


def _topk_margin(candidate_cosine: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    scores = candidate_cosine.float().masked_fill(~candidate_mask.bool(), float("-inf"))
    if int(scores.shape[1]) <= 1:
        return torch.zeros((int(scores.shape[0]),), dtype=torch.float32, device=scores.device)
    sorted_scores = torch.sort(scores, dim=1, descending=True).values
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    return torch.where(torch.isfinite(margin), margin, torch.zeros_like(margin))


def build_ulfloc_scene_matcher_pair_cache(
    *,
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    candidate_landmark_ids: torch.Tensor,
    candidate_cosine: torch.Tensor,
    keypoint_xy: torch.Tensor,
    candidate_reprojection_error: torch.Tensor,
    candidate_visible: torch.Tensor | None = None,
    query_score: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    base_gaussian_ids: torch.Tensor | None = None,
    reprojection_threshold_px: float = 4.0,
) -> dict[str, Any]:
    """Build a listwise SceneMatchNet cache from self-map top-K sparse candidates."""

    q = F.normalize(torch.as_tensor(query_desc, dtype=torch.float32), p=2, dim=-1)
    lm_all = F.normalize(torch.as_tensor(landmark_desc, dtype=torch.float32), p=2, dim=-1)
    candidate_ids = torch.as_tensor(candidate_landmark_ids, dtype=torch.long)
    cosine = torch.as_tensor(candidate_cosine, dtype=torch.float32)
    reproj = torch.as_tensor(candidate_reprojection_error, dtype=torch.float32)
    if candidate_ids.dim() != 2:
        raise ValueError("candidate_landmark_ids must be [N,K]")
    count, topk = int(candidate_ids.shape[0]), int(candidate_ids.shape[1])
    if q.shape[0] != count or cosine.shape != candidate_ids.shape or reproj.shape != candidate_ids.shape:
        raise ValueError("query_desc, candidate_cosine, and reprojection error must agree with candidate ids")
    if lm_all.dim() != 2:
        raise ValueError("landmark_desc must be [M,D]")
    if base_gaussian_ids is None:
        gaussian_ids = torch.arange(int(lm_all.shape[0]), dtype=torch.long)
    else:
        gaussian_ids = torch.as_tensor(base_gaussian_ids, dtype=torch.long).reshape(-1)
        if int(gaussian_ids.numel()) != int(lm_all.shape[0]):
            raise ValueError("base_gaussian_ids must match landmark_desc rows")
    safe_ids = candidate_ids.clamp(min=0, max=max(0, int(lm_all.shape[0]) - 1))
    candidate_desc = lm_all[safe_ids]
    visible = (
        torch.ones((count, topk), dtype=torch.bool)
        if candidate_visible is None
        else torch.as_tensor(candidate_visible, dtype=torch.bool)
    )
    candidate_mask = visible & torch.isfinite(cosine) & torch.isfinite(reproj)
    if query_score is None:
        q_score = torch.ones((count,), dtype=torch.float32)
    else:
        q_score = torch.as_tensor(query_score, dtype=torch.float32).reshape(-1)
        if int(q_score.numel()) != count:
            raise ValueError("query_score must have one value per query keypoint")
    if landmark_prior is None:
        prior = torch.ones((count, topk), dtype=torch.float32)
    else:
        raw_prior = torch.as_tensor(landmark_prior, dtype=torch.float32)
        if raw_prior.shape == (count, topk):
            prior = raw_prior
        else:
            flat_prior = raw_prior.reshape(-1)
            if int(flat_prior.numel()) == 0:
                prior = torch.ones((count, topk), dtype=torch.float32)
            else:
                prior = flat_prior[safe_ids.clamp(max=int(flat_prior.numel()) - 1)]

    episode = build_selfmap_episode_cache(
        query_id=torch.arange(count, dtype=torch.long),
        keypoint_yx=torch.stack(
            [
                torch.as_tensor(keypoint_xy, dtype=torch.float32)[:, 1],
                torch.as_tensor(keypoint_xy, dtype=torch.float32)[:, 0],
            ],
            dim=1,
        ),
        candidate_landmark_ids=candidate_ids,
        candidate_cosine=cosine,
        candidate_reprojection_error=reproj,
        candidate_visible=candidate_mask,
        reprojection_threshold_px=float(reprojection_threshold_px),
        num_landmarks=int(lm_all.shape[0]),
        false_positive_score_threshold=0.5,
    )
    labels = torch.as_tensor(episode["listwise_label"], dtype=torch.long)
    return {
        "query_desc": q.detach().cpu(),
        "base_landmark_desc": lm_all.detach().cpu(),
        "base_gaussian_id": gaussian_ids.detach().cpu(),
        "landmark_id": candidate_ids.detach().cpu(),
        "landmark_desc": candidate_desc.detach().cpu(),
        "cosine": cosine.detach().cpu(),
        "margin": _topk_margin(cosine, candidate_mask).detach().cpu(),
        "query_score": q_score.detach().cpu(),
        "landmark_prior": prior.detach().cpu(),
        "candidate_mask": candidate_mask.detach().cpu(),
        "reprojection_error": reproj.detach().cpu(),
        "label": labels.detach().cpu(),
        "metadata": {
            "format": "ulfloc_scene_matcher_listwise_pair_cache_v1",
            "topk": int(topk),
            "query_count": int(count),
            "positive_query_count": int((labels < topk).sum().item()),
            "dustbin_query_count": int((labels >= topk).sum().item()),
            "reprojection_threshold_px": float(reprojection_threshold_px),
        },
    }
