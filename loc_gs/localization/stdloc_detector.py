from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    if int(nms_radius) < 0:
        raise ValueError("nms_radius must be non-negative")
    squeeze = False
    if scores.dim() == 2:
        scores = scores.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif scores.dim() == 3:
        scores = scores.unsqueeze(0)
        squeeze = True
    if scores.dim() != 4:
        raise ValueError("scores must have shape [H,W], [1,H,W], or [B,1,H,W]")

    radius = int(nms_radius)
    if radius == 0:
        return scores.squeeze(0).squeeze(0) if squeeze else scores

    def max_pool(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    result = torch.where(max_mask, scores, zeros)
    return result.squeeze(0).squeeze(0) if squeeze else result


class StdlocKeypointDetector(nn.Module):
    """STDLoc scene-specific keypoint detector reimplemented inside loc_gs."""

    def __init__(self, in_dim: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)
            return self.sigmoid(self.cnn(feature_map))[0]
        return self.sigmoid(self.cnn(feature_map))


@torch.no_grad()
def extract_stdloc_detector_keypoints(
    feature_map: torch.Tensor,
    detector: StdlocKeypointDetector,
    max_keypoints: int = 2048,
    nms_radius: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    heat_map = detector(feature_map.float())
    if heat_map.dim() == 3:
        heat_map = heat_map[0]
    scores = simple_nms(heat_map, nms_radius).reshape(-1)
    if scores.numel() == 0:
        return (
            feature_map.new_empty((0, 2)),
            feature_map.new_empty((0,)),
        )
    topk = min(int(max_keypoints), scores.numel())
    values, ids = torch.topk(scores, k=topk)
    keep = values > 0.0
    ids = ids[keep]
    values = values[keep]
    height, width = feature_map.shape[-2:]
    y = ids // width
    x = ids % width
    keypoints = torch.stack([y.float(), x.float()], dim=-1)
    return keypoints, values
