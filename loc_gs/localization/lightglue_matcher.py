from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

_MATCHER_CACHE: dict[tuple[str, float, str], Callable] = {}


def lightglue_feature_name(dim_pipeline: str) -> str:
    pipeline = str(dim_pipeline).strip().lower()
    if pipeline.startswith("superpoint"):
        return "superpoint"
    if pipeline.startswith("aliked"):
        return "aliked"
    if pipeline.startswith("disk"):
        return "disk"
    return pipeline.split("+", 1)[0]


def make_lafs_from_yx(keypoints_yx: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    keypoints_yx = keypoints_yx.float()
    lafs = keypoints_yx.new_zeros((1, keypoints_yx.shape[0], 2, 3))
    lafs[0, :, 0, 0] = float(scale)
    lafs[0, :, 1, 1] = float(scale)
    lafs[0, :, 0, 2] = keypoints_yx[:, 1]
    lafs[0, :, 1, 2] = keypoints_yx[:, 0]
    return lafs


def _build_kornia_lightglue(feature_name: str, filter_threshold: float = 0.1):
    try:
        from kornia.feature import LightGlueMatcher
    except Exception as exc:  # pragma: no cover - dependency availability varies
        raise ImportError("Kornia LightGlueMatcher is required for LightGlue matching") from exc
    params = {"filter_threshold": float(filter_threshold)}
    return LightGlueMatcher(feature_name=feature_name, params=params)


def match_lightglue_descriptors(
    query_keypoints_yx: torch.Tensor,
    query_descriptors: torch.Tensor,
    rendered_keypoints_yx: torch.Tensor,
    rendered_descriptors: torch.Tensor,
    image_hw: Optional[tuple[int, int]] = None,
    rendered_hw: Optional[tuple[int, int]] = None,
    feature_name: str = "superpoint",
    matcher: Optional[Callable] = None,
    filter_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match two sparse descriptor sets with LightGlue/Kornia compatible API."""
    device = query_descriptors.device
    if (
        query_keypoints_yx.shape[0] < 2
        or rendered_keypoints_yx.shape[0] < 2
        or query_descriptors.numel() == 0
        or rendered_descriptors.numel() == 0
    ):
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, query_descriptors.new_empty(0)

    if matcher is None:
        cache_key = (str(feature_name), float(filter_threshold), str(query_descriptors.device))
        matcher = _MATCHER_CACHE.get(cache_key)
        if matcher is None:
            matcher = _build_kornia_lightglue(feature_name, filter_threshold=filter_threshold)
            if hasattr(matcher, "to"):
                matcher = matcher.to(query_descriptors.device)
            _MATCHER_CACHE[cache_key] = matcher
    if hasattr(matcher, "to"):
        matcher = matcher.to(query_descriptors.device)
    query_desc = F.normalize(query_descriptors.float(), p=2, dim=-1)
    rendered_desc = F.normalize(rendered_descriptors.float(), p=2, dim=-1)
    lafs_q = make_lafs_from_yx(query_keypoints_yx.to(device=query_desc.device))
    lafs_r = make_lafs_from_yx(rendered_keypoints_yx.to(device=query_desc.device))
    scores, matches = matcher(
        query_desc,
        rendered_desc.to(device=query_desc.device),
        lafs_q,
        lafs_r,
        hw1=image_hw,
        hw2=rendered_hw,
    )
    if matches.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, query_descriptors.new_empty(0)
    matches = matches.to(device=device, dtype=torch.long)
    scores = scores.reshape(-1).to(device=device, dtype=query_descriptors.dtype)
    return matches[:, 0], matches[:, 1], scores
