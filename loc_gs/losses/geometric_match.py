from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _pixel_grid_yx(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1)


def geometric_keypoint_match_loss(
    query_descs: torch.Tensor,
    query_keypoints_yx: torch.Tensor,
    query_mask: torch.Tensor,
    rendered_desc: torch.Tensor,
    valid_pixels: Optional[torch.Tensor] = None,
    locability_map: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    target_sigma_px: float = 1.0,
    locability_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Contrast teacher keypoints against rendered descriptors using geometry.

    This is intentionally not a dense SuperPoint reconstruction objective.  It
    only asks the rendered field to make geometrically corresponding keypoint
    pixels easy to retrieve, which gives the PnP stage usable correspondences
    before the pose loop becomes reliable.
    """
    B, _C, H, W = rendered_desc.shape
    P = H * W
    dtype = rendered_desc.dtype
    device = rendered_desc.device

    rendered = F.normalize(rendered_desc.float(), p=2, dim=1)
    query = F.normalize(query_descs.float(), p=2, dim=-1)
    if valid_pixels is None:
        valid = torch.ones(B, P, device=device, dtype=torch.bool)
    else:
        valid = valid_pixels.to(device=device, dtype=torch.bool).reshape(B, P)

    logits = torch.bmm(query, rendered.flatten(2))
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~valid[:, None, :], -1e4)
    log_probs = F.log_softmax(logits, dim=-1)

    grid_yx = _pixel_grid_yx(H, W, device, dtype=torch.float32)
    target_err = torch.linalg.norm(
        grid_yx.view(1, 1, P, 2) - query_keypoints_yx.float().view(B, -1, 1, 2),
        dim=-1,
    )
    local_support = target_err <= max(float(target_sigma_px) * 3.0, 1.0)
    has_target = (local_support & valid[:, None, :]).any(dim=-1)
    target_logits = -0.5 * (target_err / max(float(target_sigma_px), 1e-6)).square()
    target_logits = target_logits.masked_fill(~valid[:, None, :], -1e4)
    target = F.softmax(target_logits, dim=-1)

    valid_queries = query_mask.to(device=device, dtype=torch.bool) & has_target
    denom = valid_queries.float().sum().clamp_min(1.0)
    match_loss = -((target.detach() * log_probs).sum(dim=-1) * valid_queries.float()).sum() / denom

    with torch.no_grad():
        best = logits.argmax(dim=-1)
        pred_y = (best // W).float()
        pred_x = (best % W).float()
        pred_yx = torch.stack([pred_y, pred_x], dim=-1)
        top1_err = torch.linalg.norm(pred_yx - query_keypoints_yx.float(), dim=-1)
        top1_1px = ((top1_err <= 1.0).float() * valid_queries.float()).sum() / denom

    locability_loss = rendered_desc.new_tensor(0.0)
    if locability_map is not None and locability_weight > 0.0:
        support = (target.detach() * valid_queries.float()[:, :, None]).sum(dim=1)
        support = support / support.amax(dim=1, keepdim=True).clamp_min(1e-6)
        loc = locability_map.float()
        if loc.shape[-2:] != (H, W):
            loc = F.interpolate(loc, size=(H, W), mode="bilinear", align_corners=False)
        loc = loc.flatten(2).squeeze(1).clamp(1e-4, 1.0 - 1e-4)
        locability_loss = F.binary_cross_entropy(loc, support.clamp(0.0, 1.0))

    total = match_loss + float(locability_weight) * locability_loss
    return {
        "total": total,
        "match": match_loss,
        "locability": locability_loss,
        "top1_1px": top1_1px,
        "valid_queries": valid_queries.float().sum(),
    }
