"""Query-conditioned pair matcher for localization feedback experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from loc_gs.losses.localization_loss import project_world_to_image_yx


DEFAULT_SCALAR_DIM = 4


def scene_match_feature_dim(
    descriptor_dim: int = 256,
    scalar_dim: int = DEFAULT_SCALAR_DIM,
    include_raw_descriptors: bool = False,
) -> int:
    descriptor_blocks = 4 if include_raw_descriptors else 2
    return descriptor_blocks * int(descriptor_dim) + int(scalar_dim)


def _as_pair_scalar(
    value: torch.Tensor | None,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(count, 1, device=device, dtype=dtype)
    tensor = value.to(device=device, dtype=dtype).reshape(-1, 1)
    if tensor.shape[0] != count:
        raise ValueError(f"pair scalar has {tensor.shape[0]} rows, expected {count}")
    return tensor


def build_scene_match_pair_features(
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    *,
    cosine: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    calibrated_prior: torch.Tensor | None = None,
    extra_scalar_features: torch.Tensor | None = None,
    include_raw_descriptors: bool = False,
) -> torch.Tensor:
    """Build fixed pair features for a query-conditioned 2D-3D matcher.

    The default representation intentionally keeps the STDLoc descriptor space
    intact: descriptor product and absolute difference carry the pair relation,
    while scalar channels expose cosine score, ambiguity margin, landmark prior,
    and optional calibrated feedback.
    """

    if query_desc.shape != landmark_desc.shape:
        raise ValueError("query_desc and landmark_desc must have the same shape")
    if query_desc.dim() != 2:
        raise ValueError("query_desc and landmark_desc must be [N, D]")
    count = int(query_desc.shape[0])
    q = F.normalize(query_desc.float(), p=2, dim=-1)
    d = F.normalize(landmark_desc.float(), p=2, dim=-1)
    blocks = []
    if include_raw_descriptors:
        blocks.extend([q, d])
    blocks.extend([q * d, (q - d).abs()])
    if cosine is None:
        cosine = (q * d).sum(dim=-1)
    scalars = [
        _as_pair_scalar(cosine, count, device=q.device, dtype=q.dtype),
        _as_pair_scalar(margin, count, device=q.device, dtype=q.dtype),
        _as_pair_scalar(landmark_prior, count, device=q.device, dtype=q.dtype),
        _as_pair_scalar(calibrated_prior, count, device=q.device, dtype=q.dtype),
    ]
    if extra_scalar_features is not None:
        extra = extra_scalar_features.to(device=q.device, dtype=q.dtype)
        if extra.dim() == 1:
            extra = extra.reshape(-1, 1)
        if extra.shape[0] != count:
            raise ValueError(f"extra_scalar_features has {extra.shape[0]} rows, expected {count}")
        scalars.append(extra)
    return torch.cat([*blocks, *scalars], dim=-1)


class SceneMatchNet(nn.Module):
    """Small MLP that predicts whether a 2D-3D pair is PnP-useful."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        scalar_dim: int = DEFAULT_SCALAR_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        include_raw_descriptors: bool = False,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")
        self.config = {
            "descriptor_dim": int(descriptor_dim),
            "scalar_dim": int(scalar_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "include_raw_descriptors": bool(include_raw_descriptors),
        }
        input_dim = scene_match_feature_dim(
            descriptor_dim=descriptor_dim,
            scalar_dim=scalar_dim,
            include_raw_descriptors=include_raw_descriptors,
        )
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(0, int(num_layers) - 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features.float()).squeeze(-1)

    def forward(
        self,
        query_desc: torch.Tensor,
        landmark_desc: torch.Tensor,
        *,
        cosine: torch.Tensor | None = None,
        margin: torch.Tensor | None = None,
        landmark_prior: torch.Tensor | None = None,
        calibrated_prior: torch.Tensor | None = None,
        extra_scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = build_scene_match_pair_features(
            query_desc,
            landmark_desc,
            cosine=cosine,
            margin=margin,
            landmark_prior=landmark_prior,
            calibrated_prior=calibrated_prior,
            extra_scalar_features=extra_scalar_features,
            include_raw_descriptors=bool(self.config["include_raw_descriptors"]),
        )
        return self.score_features(features)


def load_scene_matcher(path: str | Path, device: torch.device | str = "cpu") -> SceneMatchNet:
    checkpoint: Any = torch.load(Path(path), map_location=device)
    if isinstance(checkpoint, SceneMatchNet):
        return checkpoint.to(device).eval()
    if not isinstance(checkpoint, dict):
        raise ValueError("scene matcher checkpoint must be a dict or SceneMatchNet")
    config = checkpoint.get("config", {})
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError("scene matcher checkpoint is missing a state_dict")
    model = SceneMatchNet(**config).to(device)
    model.load_state_dict(state_dict)
    return model.eval()


@torch.no_grad()
def score_scene_match_pairs(
    matcher: SceneMatchNet,
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    *,
    cosine: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    calibrated_prior: torch.Tensor | None = None,
    extra_scalar_features: torch.Tensor | None = None,
) -> torch.Tensor:
    matcher.eval()
    logits = matcher(
        query_desc,
        landmark_desc,
        cosine=cosine,
        margin=margin,
        landmark_prior=landmark_prior,
        calibrated_prior=calibrated_prior,
        extra_scalar_features=extra_scalar_features,
    )
    return logits.float()


def best_pair_per_query_indices(
    query_ids: torch.Tensor,
    pair_scores: torch.Tensor,
    *,
    num_queries: int | None = None,
) -> torch.Tensor:
    """Return indices of the highest-scoring candidate for each query id."""

    q = query_ids.long().reshape(-1)
    scores = pair_scores.float().reshape(-1)
    if q.numel() != scores.numel():
        raise ValueError("query_ids and pair_scores must have the same length")
    if q.numel() == 0:
        return q.new_empty(0)
    if num_queries is None:
        num_queries = int(q.max().item()) + 1
    best_score = scores.new_full((int(num_queries),), float("-inf"))
    best_index = q.new_full((int(num_queries),), -1)
    for idx in range(int(q.numel())):
        qi = int(q[idx].item())
        if qi < 0 or qi >= int(num_queries):
            continue
        score = scores[idx]
        if score > best_score[qi]:
            best_score[qi] = score
            best_index[qi] = idx
    keep = best_index[best_index >= 0]
    if keep.numel() <= 1:
        return keep
    return keep[torch.argsort(scores[keep], descending=True)]


def _camera_depth(points: torch.Tensor, pose_w2c: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)
    cam = (pose_w2c.to(device=points.device, dtype=points.dtype) @ pts_h.T).T
    return cam[:, 2]


def _sample_scalar_map_bilinear(value_map: torch.Tensor, yx: torch.Tensor) -> torch.Tensor:
    if value_map.dim() == 2:
        value_map = value_map.unsqueeze(0)
    if value_map.dim() != 3 or value_map.shape[0] != 1:
        value_map = value_map.reshape(1, int(value_map.shape[-2]), int(value_map.shape[-1]))
    _, h, w = value_map.shape
    y = yx[:, 0].float()
    x = yx[:, 1].float()
    x_norm = 2.0 * x / max(w - 1, 1) - 1.0
    y_norm = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        value_map.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1)


@torch.no_grad()
def label_scene_match_pairs(
    *,
    query_yx: torch.Tensor,
    landmark_xyz: torch.Tensor,
    q_ids: torch.Tensor,
    lm_ids: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    reprojection_threshold_px: float,
    depth_map: torch.Tensor | None = None,
    alpha_map: torch.Tensor | None = None,
    depth_abs_tolerance: float = 0.25,
    depth_rel_tolerance: float = 0.02,
    alpha_threshold: float = 0.05,
) -> torch.Tensor:
    """Return PnP-useful labels for top-k query-landmark candidates."""

    if q_ids.numel() == 0 or lm_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=landmark_xyz.device)
    device = landmark_xyz.device
    q = q_ids.to(device=device, dtype=torch.long).reshape(-1)
    lm = lm_ids.to(device=device, dtype=torch.long).reshape(-1)
    pts = landmark_xyz[lm]
    proj_yx, valid_z = project_world_to_image_yx(
        pts.unsqueeze(0),
        pose_w2c.to(device=device, dtype=torch.float32).unsqueeze(0),
        K.to(device=device, dtype=torch.float32),
    )
    proj = proj_yx[0]
    err = torch.linalg.norm(proj - query_yx.to(device=device, dtype=torch.float32)[q], dim=-1)
    valid = valid_z[0] & torch.isfinite(err)
    visibility_map = depth_map if depth_map is not None else alpha_map
    if visibility_map is not None:
        h = int(visibility_map.shape[-2])
        w = int(visibility_map.shape[-1])
        in_frame = (proj[:, 0] >= 0.0) & (proj[:, 0] <= h - 1) & (proj[:, 1] >= 0.0) & (proj[:, 1] <= w - 1)
        valid = valid & in_frame
        if depth_map is not None:
            sampled_depth = _sample_scalar_map_bilinear(depth_map.to(device=device), proj)
            z = _camera_depth(pts.float(), pose_w2c.to(device=device, dtype=torch.float32))
            depth_tol = max(float(depth_abs_tolerance), 0.0) + max(float(depth_rel_tolerance), 0.0) * z.abs()
            depth_ok = (sampled_depth > 0.0) & torch.isfinite(sampled_depth) & ((sampled_depth - z).abs() <= depth_tol)
            valid = valid & depth_ok
        if alpha_map is not None:
            sampled_alpha = _sample_scalar_map_bilinear(alpha_map.to(device=device), proj)
            valid = valid & torch.isfinite(sampled_alpha) & (sampled_alpha >= float(alpha_threshold))
    return valid & (err <= float(reprojection_threshold_px))
