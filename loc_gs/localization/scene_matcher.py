"""Query-conditioned pair matcher for localization feedback experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from loc_gs.losses.localization_loss import project_world_to_image_yx


DEFAULT_SCALAR_DIM = 4
DEFAULT_LISTWISE_BASE_SCALAR_DIM = 4
DEFAULT_LISTWISE_EXTRA_FEATURES = "query_score"
LISTWISE_EXTRA_FEATURE_DIMS = {
    "none": 0,
    "query_score": 1,
    "query_score_rank_gap": 3,
}
DEFAULT_LISTWISE_SCALAR_DIM = DEFAULT_LISTWISE_BASE_SCALAR_DIM + LISTWISE_EXTRA_FEATURE_DIMS[
    DEFAULT_LISTWISE_EXTRA_FEATURES
]


def scene_match_feature_dim(
    descriptor_dim: int = 256,
    scalar_dim: int = DEFAULT_SCALAR_DIM,
    include_raw_descriptors: bool = False,
) -> int:
    descriptor_blocks = 4 if include_raw_descriptors else 2
    return descriptor_blocks * int(descriptor_dim) + int(scalar_dim)


def scene_match_listwise_feature_dim(
    descriptor_dim: int = 256,
    scalar_dim: int = DEFAULT_LISTWISE_SCALAR_DIM,
    include_raw_descriptors: bool = False,
) -> int:
    return scene_match_feature_dim(
        descriptor_dim=descriptor_dim,
        scalar_dim=scalar_dim,
        include_raw_descriptors=include_raw_descriptors,
    )


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


def _as_listwise_scalar(
    value: torch.Tensor | None,
    batch: int,
    topk: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch, topk, 1, device=device, dtype=dtype)
    tensor = value.to(device=device, dtype=dtype)
    if tensor.dim() == 1:
        if tensor.shape[0] != batch:
            raise ValueError(f"listwise scalar has {tensor.shape[0]} rows, expected {batch}")
        tensor = tensor.view(batch, 1, 1).expand(batch, topk, 1)
    elif tensor.dim() == 2:
        if tensor.shape == (batch, topk):
            tensor = tensor.unsqueeze(-1)
        elif tensor.shape[0] == batch:
            tensor = tensor[:, None, :].expand(batch, topk, tensor.shape[-1])
        else:
            raise ValueError(f"listwise scalar has shape {tuple(tensor.shape)}, expected [{batch}, {topk}]")
    elif tensor.dim() == 3:
        if tensor.shape[0] != batch or tensor.shape[1] != topk:
            raise ValueError(f"listwise scalar has shape {tuple(tensor.shape)}, expected [{batch}, {topk}, C]")
    else:
        raise ValueError("listwise scalar must be [B], [B,K], [B,C], or [B,K,C]")
    return tensor


def listwise_extra_feature_dim(mode: str = DEFAULT_LISTWISE_EXTRA_FEATURES) -> int:
    mode = str(mode)
    if mode not in LISTWISE_EXTRA_FEATURE_DIMS:
        raise ValueError(f"unsupported listwise extra feature mode: {mode}")
    return int(LISTWISE_EXTRA_FEATURE_DIMS[mode])


def build_scene_match_listwise_extra_features(
    mode: str = DEFAULT_LISTWISE_EXTRA_FEATURES,
    *,
    query_score: torch.Tensor | None = None,
    cosine: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build derived per-candidate scalar context for listwise SceneMatchNet."""
    mode = str(mode)
    extra_dim = listwise_extra_feature_dim(mode)
    if extra_dim == 0:
        return None
    if cosine is None:
        if query_score is None:
            raise ValueError("cosine or query_score is required to infer listwise extra feature shape")
        score = query_score
        if score.dim() == 1:
            raise ValueError("cosine is required when query_score is one-dimensional")
        batch, topk = int(score.shape[0]), int(score.shape[1])
        device = score.device
        dtype = score.dtype
        cosine_tensor = torch.zeros(batch, topk, device=device, dtype=dtype)
    else:
        cosine_tensor = cosine.float()
        if cosine_tensor.dim() != 2:
            raise ValueError("cosine must be [B,K] for listwise extra features")
        batch, topk = int(cosine_tensor.shape[0]), int(cosine_tensor.shape[1])
        device = cosine_tensor.device
        dtype = cosine_tensor.dtype
    score_feature = _as_listwise_scalar(
        query_score,
        batch,
        topk,
        device=device,
        dtype=dtype,
    )
    if mode == "query_score":
        return score_feature
    if candidate_mask is None:
        mask = torch.ones(batch, topk, dtype=torch.bool, device=device)
    else:
        mask = candidate_mask.to(device=device, dtype=torch.bool)
        if mask.shape != (batch, topk):
            raise ValueError(f"candidate_mask must have shape {(batch, topk)}")
    if topk <= 1:
        rank = torch.zeros(batch, topk, 1, device=device, dtype=dtype)
    else:
        rank = torch.linspace(0.0, 1.0, topk, device=device, dtype=dtype).view(1, topk, 1)
        rank = rank.expand(batch, topk, 1)
    masked_cosine = cosine_tensor.masked_fill(~mask, float("-inf"))
    best = masked_cosine.amax(dim=1)
    best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
    gap = (best[:, None] - cosine_tensor).masked_fill(~mask, 0.0).unsqueeze(-1)
    return torch.cat([score_feature, rank, gap], dim=-1)


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


def build_scene_match_listwise_features(
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
    """Build candidate-list features for one query keypoint and its top-K landmarks."""
    if query_desc.dim() != 2:
        raise ValueError("query_desc must be [B, D]")
    if landmark_desc.dim() != 3:
        raise ValueError("landmark_desc must be [B, K, D]")
    if query_desc.shape[0] != landmark_desc.shape[0] or query_desc.shape[1] != landmark_desc.shape[2]:
        raise ValueError("query_desc [B,D] and landmark_desc [B,K,D] dimensions must agree")
    batch = int(query_desc.shape[0])
    topk = int(landmark_desc.shape[1])
    q = F.normalize(query_desc.float(), p=2, dim=-1)
    d = F.normalize(landmark_desc.float(), p=2, dim=-1)
    q_expanded = q[:, None, :].expand(batch, topk, q.shape[-1])
    blocks = []
    if include_raw_descriptors:
        blocks.extend([q_expanded, d])
    blocks.extend([q_expanded * d, (q_expanded - d).abs()])
    if cosine is None:
        cosine = (q_expanded * d).sum(dim=-1)
    scalars = [
        _as_listwise_scalar(cosine, batch, topk, device=q.device, dtype=q.dtype),
        _as_listwise_scalar(margin, batch, topk, device=q.device, dtype=q.dtype),
        _as_listwise_scalar(landmark_prior, batch, topk, device=q.device, dtype=q.dtype),
        _as_listwise_scalar(calibrated_prior, batch, topk, device=q.device, dtype=q.dtype),
    ]
    if extra_scalar_features is not None:
        scalars.append(_as_listwise_scalar(extra_scalar_features, batch, topk, device=q.device, dtype=q.dtype))
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
            "model_type": "pairwise",
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


class SceneMatchListwiseNet(nn.Module):
    """Listwise top-K matcher with an explicit dustbin class."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        scalar_dim: int | None = None,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        include_raw_descriptors: bool = False,
        listwise_extra_features: str = DEFAULT_LISTWISE_EXTRA_FEATURES,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")
        if scalar_dim is None:
            scalar_dim = DEFAULT_LISTWISE_BASE_SCALAR_DIM + listwise_extra_feature_dim(listwise_extra_features)
        self.config = {
            "model_type": "listwise",
            "descriptor_dim": int(descriptor_dim),
            "scalar_dim": int(scalar_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "include_raw_descriptors": bool(include_raw_descriptors),
            "listwise_extra_features": str(listwise_extra_features),
        }
        input_dim = scene_match_listwise_feature_dim(
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
        self.candidate_encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.candidate_projection = nn.Linear(dim, int(hidden_dim)) if dim != int(hidden_dim) else nn.Identity()
        self.candidate_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.dustbin_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

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
        query_score: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if extra_scalar_features is None and int(self.config["scalar_dim"]) > DEFAULT_LISTWISE_BASE_SCALAR_DIM:
            if cosine is None:
                q = F.normalize(query_desc.float(), p=2, dim=-1)
                d = F.normalize(landmark_desc.float(), p=2, dim=-1)
                cosine_for_extra = (q[:, None, :] * d).sum(dim=-1)
            else:
                cosine_for_extra = cosine
            extra_scalar_features = build_scene_match_listwise_extra_features(
                str(self.config.get("listwise_extra_features", DEFAULT_LISTWISE_EXTRA_FEATURES)),
                query_score=query_score,
                cosine=cosine_for_extra,
                candidate_mask=candidate_mask,
            )
        features = build_scene_match_listwise_features(
            query_desc,
            landmark_desc,
            cosine=cosine,
            margin=margin,
            landmark_prior=landmark_prior,
            calibrated_prior=calibrated_prior,
            extra_scalar_features=extra_scalar_features,
            include_raw_descriptors=bool(self.config["include_raw_descriptors"]),
        )
        encoded = self.candidate_projection(self.candidate_encoder(features.float()))
        batch, topk, hidden = encoded.shape
        if candidate_mask is None:
            mask = torch.ones(batch, topk, dtype=torch.bool, device=encoded.device)
        else:
            mask = candidate_mask.to(device=encoded.device, dtype=torch.bool)
            if mask.shape != (batch, topk):
                raise ValueError(f"candidate_mask must have shape {(batch, topk)}")
        encoded_masked = encoded.masked_fill(~mask[..., None], 0.0)
        denom = mask.sum(dim=1).clamp_min(1).to(device=encoded.device, dtype=encoded.dtype).view(batch, 1)
        pooled_mean = encoded_masked.sum(dim=1) / denom
        pooled_max = encoded.masked_fill(~mask[..., None], float("-inf")).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        context = torch.cat(
            [
                encoded,
                pooled_mean[:, None, :].expand(batch, topk, hidden),
                pooled_max[:, None, :].expand(batch, topk, hidden),
            ],
            dim=-1,
        )
        candidate_logits = self.candidate_head(context).squeeze(-1).masked_fill(~mask, float("-inf"))
        dustbin_logits = self.dustbin_head(torch.cat([pooled_mean, pooled_max], dim=-1)).squeeze(-1)
        return torch.cat([candidate_logits, dustbin_logits[:, None]], dim=-1)


def load_scene_matcher(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint: Any = torch.load(Path(path), map_location=device)
    if isinstance(checkpoint, nn.Module):
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
    model_type = str(config.get("model_type", "pairwise"))
    if model_type == "listwise":
        model = SceneMatchListwiseNet(**{k: v for k, v in config.items() if k != "model_type"}).to(device)
    else:
        model = SceneMatchNet(**{k: v for k, v in config.items() if k != "model_type"}).to(device)
    model.load_state_dict(state_dict)
    return model.eval()


@torch.no_grad()
def score_scene_match_pairs(
    matcher: nn.Module,
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


@torch.no_grad()
def score_scene_match_candidates(
    matcher: nn.Module,
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    *,
    cosine: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    calibrated_prior: torch.Tensor | None = None,
    extra_scalar_features: torch.Tensor | None = None,
    query_score: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if str(getattr(matcher, "config", {}).get("model_type", "pairwise")) != "listwise":
        raise ValueError("score_scene_match_candidates requires a listwise scene matcher")
    matcher.eval()
    logits = matcher(
        query_desc,
        landmark_desc,
        cosine=cosine,
        margin=margin,
        landmark_prior=landmark_prior,
        calibrated_prior=calibrated_prior,
        extra_scalar_features=extra_scalar_features,
        query_score=query_score,
        candidate_mask=candidate_mask,
    )
    return logits.float()


@torch.no_grad()
def select_scene_match_listwise_candidates(
    matcher: nn.Module,
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    q_ids: torch.Tensor,
    lm_ids: torch.Tensor,
    *,
    cosine: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    calibrated_prior: torch.Tensor | None = None,
    query_score: torch.Tensor | None = None,
    drop_dustbin: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one candidate per query from flat top-K matches, allowing dustbin rejection."""
    if str(getattr(matcher, "config", {}).get("model_type", "pairwise")) != "listwise":
        raise ValueError("select_scene_match_listwise_candidates requires a listwise scene matcher")
    q = q_ids.long().reshape(-1)
    lm = lm_ids.long().reshape(-1)
    if q.numel() != lm.numel():
        raise ValueError("q_ids and lm_ids must have the same length")
    device = query_desc.device
    if q.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), query_desc.new_empty(0)
    q = q.to(device=device)
    lm = lm.to(device=device)
    unique_q = torch.unique(q, sorted=True)
    counts = torch.stack([(q == qi).sum() for qi in unique_q], dim=0)
    topk = int(counts.max().item())
    batch = int(unique_q.numel())
    candidate_lm = torch.zeros(batch, topk, dtype=torch.long, device=device)
    candidate_mask = torch.zeros(batch, topk, dtype=torch.bool, device=device)
    flat_indices = torch.full((batch, topk), -1, dtype=torch.long, device=device)

    def _fill_scalar(value: torch.Tensor | None, fill: float = 0.0) -> torch.Tensor | None:
        if value is None:
            return None
        src = value.to(device=device).reshape(-1)
        out = torch.full((batch, topk), float(fill), dtype=src.dtype, device=device)
        for row, qi in enumerate(unique_q):
            local = torch.where(q == qi)[0][:topk]
            if local.numel() > 0:
                out[row, : int(local.numel())] = src[local]
        return out

    for row, qi in enumerate(unique_q):
        local = torch.where(q == qi)[0][:topk]
        if local.numel() == 0:
            continue
        count = int(local.numel())
        candidate_lm[row, :count] = lm[local]
        candidate_mask[row, :count] = True
        flat_indices[row, :count] = local
    margin_group = None
    if margin is not None:
        margin_flat = margin.to(device=device).reshape(-1)
        margin_group = torch.zeros(batch, dtype=margin_flat.dtype, device=device)
        for row, qi in enumerate(unique_q):
            local = torch.where(q == qi)[0]
            if local.numel() > 0:
                margin_group[row] = margin_flat[local[0]]
    query_score_group = None
    if query_score is not None:
        query_score = query_score.to(device=device).reshape(-1)
        if query_score.numel() == int(query_desc.shape[0]):
            query_score_group = query_score[unique_q]
        elif query_score.numel() == int(q.numel()):
            query_score_group = torch.zeros(batch, dtype=query_score.dtype, device=device)
            for row, qi in enumerate(unique_q):
                local = torch.where(q == qi)[0]
                if local.numel() > 0:
                    query_score_group[row] = query_score[local[0]]
        else:
            raise ValueError("query_score must be per-query or per-flat-candidate")
    logits = score_scene_match_candidates(
        matcher,
        query_desc[unique_q],
        landmark_desc[candidate_lm],
        cosine=_fill_scalar(cosine),
        margin=margin_group,
        landmark_prior=_fill_scalar(landmark_prior),
        calibrated_prior=_fill_scalar(calibrated_prior),
        query_score=query_score_group,
        candidate_mask=candidate_mask,
    )
    candidate_logits = logits[:, :topk]
    dustbin_logits = logits[:, topk]
    best_logits, best_cols = candidate_logits.max(dim=1)
    rows = torch.arange(batch, dtype=torch.long, device=device)
    selected_flat = flat_indices[rows, best_cols]
    selected_logits = best_logits
    if not bool(drop_dustbin):
        selected_logits = best_logits - dustbin_logits
    keep = (selected_flat >= 0) & torch.isfinite(best_logits)
    if bool(drop_dustbin):
        keep = keep & (best_logits > dustbin_logits)
    if not bool(keep.any()):
        return torch.empty(0, dtype=torch.long, device=device), query_desc.new_empty(0)
    selected_flat = selected_flat[keep]
    selected_logits = selected_logits[keep]
    order = torch.argsort(selected_logits, descending=True)
    return selected_flat[order], selected_logits[order]


@torch.no_grad()
def score_scene_match_listwise_flat_candidates(
    matcher: nn.Module,
    query_desc: torch.Tensor,
    landmark_desc: torch.Tensor,
    q_ids: torch.Tensor,
    lm_ids: torch.Tensor,
    *,
    cosine: torch.Tensor | None = None,
    margin: torch.Tensor | None = None,
    landmark_prior: torch.Tensor | None = None,
    calibrated_prior: torch.Tensor | None = None,
    query_score: torch.Tensor | None = None,
    drop_dustbin: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score flat top-K candidates without hard-selecting a matcher winner.

    The returned logits are aligned with the returned flat indices and can be
    blended with descriptor scores before the final one-per-query PROSAC
    ordering is chosen.  This keeps the listwise readout as a soft
    localization-feedback signal instead of an implicit hard branch selector.
    """
    if str(getattr(matcher, "config", {}).get("model_type", "pairwise")) != "listwise":
        raise ValueError("score_scene_match_listwise_flat_candidates requires a listwise scene matcher")
    q = q_ids.long().reshape(-1)
    lm = lm_ids.long().reshape(-1)
    if q.numel() != lm.numel():
        raise ValueError("q_ids and lm_ids must have the same length")
    device = query_desc.device
    if q.numel() == 0:
        return query_desc.new_empty(0), torch.empty(0, dtype=torch.long, device=device)
    q = q.to(device=device)
    lm = lm.to(device=device)
    original_flat = torch.arange(q.numel(), dtype=torch.long, device=device)
    sort_key = q * (q.numel() + 1) + original_flat
    order = torch.argsort(sort_key)
    q_sorted = q[order]
    unique_q, counts = torch.unique_consecutive(q_sorted, return_counts=True)
    topk = int(counts.max().item())
    batch = int(unique_q.numel())
    offsets = torch.arange(topk, dtype=torch.long, device=device).view(1, topk)
    starts = (torch.cumsum(counts, dim=0) - counts).view(batch, 1)
    candidate_mask = offsets < counts.view(batch, 1)
    sorted_positions = (starts + offsets).clamp(max=int(q.numel()) - 1)
    flat_indices = order[sorted_positions].masked_fill(~candidate_mask, -1)
    safe_flat_indices = flat_indices.clamp_min(0)
    candidate_lm = lm[safe_flat_indices].masked_fill(~candidate_mask, 0)

    def _fill_scalar(value: torch.Tensor | None, fill: float = 0.0) -> torch.Tensor | None:
        if value is None:
            return None
        src = value.to(device=device).reshape(-1)
        out = src[safe_flat_indices]
        return out.masked_fill(~candidate_mask, float(fill))

    margin_group = None
    if margin is not None:
        margin_flat = margin.to(device=device).reshape(-1)
        margin_group = margin_flat[safe_flat_indices[:, 0]]

    query_score_group = None
    if query_score is not None:
        query_score = query_score.to(device=device).reshape(-1)
        if query_score.numel() == int(query_desc.shape[0]):
            query_score_group = query_score[unique_q]
        elif query_score.numel() == int(q.numel()):
            query_score_group = query_score[safe_flat_indices[:, 0]]
        else:
            raise ValueError("query_score must be per-query or per-flat-candidate")

    logits = score_scene_match_candidates(
        matcher,
        query_desc[unique_q],
        landmark_desc[candidate_lm],
        cosine=_fill_scalar(cosine),
        margin=margin_group,
        landmark_prior=_fill_scalar(landmark_prior),
        calibrated_prior=_fill_scalar(calibrated_prior),
        query_score=query_score_group,
        candidate_mask=candidate_mask,
    )
    candidate_logits = logits[:, :topk]
    dustbin_logits = logits[:, topk]
    score = candidate_logits - dustbin_logits[:, None]
    valid = candidate_mask & torch.isfinite(score)
    if bool(drop_dustbin):
        valid = valid & (candidate_logits > dustbin_logits[:, None])
    if not bool(valid.any()):
        return query_desc.new_empty(0), torch.empty(0, dtype=torch.long, device=device)
    keep_indices = flat_indices[valid]
    keep_logits = score[valid]
    out_order = torch.argsort(keep_indices)
    return keep_logits[out_order], keep_indices[out_order]


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
def label_scene_match_topk_candidates(
    *,
    query_yx: torch.Tensor,
    landmark_xyz: torch.Tensor,
    lm_topk: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    reprojection_threshold_px: float,
    depth_map: torch.Tensor | None = None,
    alpha_map: torch.Tensor | None = None,
    depth_abs_tolerance: float = 0.25,
    depth_rel_tolerance: float = 0.02,
    alpha_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return listwise labels where K denotes dustbin/no positive candidate."""
    if lm_topk.dim() != 2:
        raise ValueError("lm_topk must be [num_queries, topk]")
    device = landmark_xyz.device
    lm = lm_topk.to(device=device, dtype=torch.long)
    num_queries, topk = int(lm.shape[0]), int(lm.shape[1])
    dustbin = topk
    if num_queries == 0 or topk == 0:
        return (
            torch.full((num_queries,), dustbin, dtype=torch.long, device=device),
            torch.zeros(num_queries, topk, dtype=torch.bool, device=device),
            torch.empty(num_queries, topk, dtype=torch.float32, device=device),
        )
    pts = landmark_xyz[lm.reshape(-1)]
    proj_yx, valid_z = project_world_to_image_yx(
        pts.view(1, -1, 3),
        pose_w2c.to(device=device, dtype=torch.float32).unsqueeze(0),
        K.to(device=device, dtype=torch.float32),
    )
    proj = proj_yx[0].view(num_queries, topk, 2)
    valid = valid_z[0].view(num_queries, topk)
    query = query_yx.to(device=device, dtype=torch.float32).view(num_queries, 1, 2)
    errors = torch.linalg.norm(proj - query, dim=-1)
    valid = valid & torch.isfinite(errors)
    visibility_map = depth_map if depth_map is not None else alpha_map
    if visibility_map is not None:
        h = int(visibility_map.shape[-2])
        w = int(visibility_map.shape[-1])
        in_frame = (proj[..., 0] >= 0.0) & (proj[..., 0] <= h - 1) & (proj[..., 1] >= 0.0) & (proj[..., 1] <= w - 1)
        valid = valid & in_frame
        proj_flat = proj.reshape(-1, 2)
        pts_flat = pts
        if depth_map is not None:
            sampled_depth = _sample_scalar_map_bilinear(depth_map.to(device=device), proj_flat).view(num_queries, topk)
            z = _camera_depth(pts_flat.float(), pose_w2c.to(device=device, dtype=torch.float32)).view(num_queries, topk)
            depth_tol = max(float(depth_abs_tolerance), 0.0) + max(float(depth_rel_tolerance), 0.0) * z.abs()
            depth_ok = (sampled_depth > 0.0) & torch.isfinite(sampled_depth) & ((sampled_depth - z).abs() <= depth_tol)
            valid = valid & depth_ok
        if alpha_map is not None:
            sampled_alpha = _sample_scalar_map_bilinear(alpha_map.to(device=device), proj_flat).view(num_queries, topk)
            valid = valid & torch.isfinite(sampled_alpha) & (sampled_alpha >= float(alpha_threshold))
    positive = valid & (errors <= float(reprojection_threshold_px))
    masked_errors = errors.masked_fill(~positive, float("inf"))
    best_errors, labels = masked_errors.min(dim=1)
    labels = labels.long()
    labels[~torch.isfinite(best_errors)] = dustbin
    return labels, valid, errors


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
