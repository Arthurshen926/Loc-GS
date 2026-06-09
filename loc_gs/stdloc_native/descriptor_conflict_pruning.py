from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DescriptorConflictScores:
    nearest_source_cosine: torch.Tensor
    nearest_source_distance_m: torch.Tensor
    nearest_source_gid: torch.Tensor
    conflict_score: torch.Tensor
    source_mask: torch.Tensor


@dataclass(frozen=True)
class DescriptorConflictPruneResult:
    pruned_sampled_idx: torch.Tensor
    pruned_features: torch.Tensor
    keep_mask: torch.Tensor
    scores: DescriptorConflictScores
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PairwiseDescriptorConflictEdge:
    src_pos: int
    dst_pos: int
    src_gid: int
    dst_gid: int
    cosine: float
    distance_m: float
    score: float


def _as_long_vector(values: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long).reshape(-1).cpu()
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return tensor


def _as_feature_matrix(values: torch.Tensor, *, rows: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError("features must have shape [N, C]")
    if int(tensor.shape[0]) != int(rows):
        raise ValueError(f"features row count {tensor.shape[0]} does not match sampled_idx length {rows}")
    if int(tensor.shape[1]) <= 0:
        raise ValueError("features must have at least one channel")
    return tensor.cpu()


def _as_xyz(values: torch.Tensor, *, min_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).cpu()
    if tensor.ndim != 2 or int(tensor.shape[1]) != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if int(tensor.shape[0]) < int(min_size):
        raise ValueError(f"xyz has {tensor.shape[0]} rows but sampled_idx references {min_size - 1}")
    return tensor


def compute_descriptor_conflict_scores(
    *,
    sampled_idx: torch.Tensor,
    features: torch.Tensor,
    xyz: torch.Tensor,
    source_anchor_idx: torch.Tensor,
    min_cosine: float,
    min_spatial_distance_m: float,
    chunk_size: int = 2048,
    source_chunk_size: int = 8192,
    device: str | torch.device | None = None,
) -> DescriptorConflictScores:
    """Score selected landmarks that can steal top-1 matches from source anchors.

    A non-source selected landmark is considered risky when its fused descriptor
    is close to a source-anchor descriptor while its 3D position is far away.
    This directly targets the top-1 nearest-neighbor failure mode caused by
    adding many non-source landmarks to an otherwise safe ULF source set.
    """

    sampled = _as_long_vector(sampled_idx, name="sampled_idx")
    feats = _as_feature_matrix(features, rows=int(sampled.numel()))
    max_gid = int(sampled.max().item())
    points = _as_xyz(xyz, min_size=max_gid + 1)
    anchors = set(int(v) for v in _as_long_vector(source_anchor_idx, name="source_anchor_idx").tolist())
    source_mask = torch.tensor([int(gid) in anchors for gid in sampled.tolist()], dtype=torch.bool)
    source_positions = torch.where(source_mask)[0]
    if int(source_positions.numel()) == 0:
        raise ValueError("no source_anchor_idx entries are present in sampled_idx")

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = max(1, int(chunk_size))
    source_chunk = max(1, int(source_chunk_size))
    feature_device = F.normalize(feats.to(dev), dim=1)
    source_features = feature_device[source_positions.to(dev)]
    source_gids = sampled[source_positions]
    source_xyz = points[source_gids].to(dev)

    nearest_cosine = torch.full((int(sampled.numel()),), -float("inf"), dtype=torch.float32)
    nearest_source_pos = torch.full((int(sampled.numel()),), -1, dtype=torch.long)
    for start in range(0, int(sampled.numel()), chunk):
        end = min(start + chunk, int(sampled.numel()))
        query_features = feature_device[start:end]
        best_values = torch.full((end - start,), -float("inf"), dtype=torch.float32, device=dev)
        best_positions = torch.full((end - start,), -1, dtype=torch.long, device=dev)
        for source_start in range(0, int(source_features.shape[0]), source_chunk):
            source_end = min(source_start + source_chunk, int(source_features.shape[0]))
            sims = query_features @ source_features[source_start:source_end].T
            local_values, local_pos = sims.max(dim=1)
            better = local_values > best_values
            best_values[better] = local_values[better]
            best_positions[better] = local_pos[better].to(torch.long) + int(source_start)
        nearest_cosine[start:end] = best_values.detach().cpu()
        nearest_source_pos[start:end] = best_positions.detach().cpu()

    nearest_source_gid = source_gids[nearest_source_pos.clamp_min(0)]
    sampled_xyz = points[sampled]
    nearest_xyz = points[nearest_source_gid]
    nearest_distance = torch.linalg.norm(sampled_xyz - nearest_xyz, dim=1).to(torch.float32)

    cos_margin = (nearest_cosine - float(min_cosine)).clamp_min(0.0)
    distance_ok = nearest_distance >= float(min_spatial_distance_m)
    conflict_score = cos_margin * distance_ok.to(torch.float32)
    conflict_score[source_mask] = 0.0
    nearest_cosine[source_mask] = torch.maximum(nearest_cosine[source_mask], torch.ones_like(nearest_cosine[source_mask]))
    nearest_distance[source_mask] = 0.0
    nearest_source_gid[source_mask] = sampled[source_mask]

    return DescriptorConflictScores(
        nearest_source_cosine=nearest_cosine.cpu(),
        nearest_source_distance_m=nearest_distance.cpu(),
        nearest_source_gid=nearest_source_gid.cpu(),
        conflict_score=conflict_score.cpu(),
        source_mask=source_mask.cpu(),
    )


def prune_descriptor_conflicts(
    *,
    sampled_idx: torch.Tensor,
    features: torch.Tensor,
    xyz: torch.Tensor,
    source_anchor_idx: torch.Tensor,
    protect_scores: torch.Tensor | None = None,
    min_protect_score: float = 0.0,
    min_cosine: float = 0.95,
    min_spatial_distance_m: float = 1.0,
    max_prune_fraction: float = 1.0,
    max_prune_count: int = 0,
    min_keep_count: int = 0,
    chunk_size: int = 2048,
    source_chunk_size: int = 8192,
    device: str | torch.device | None = None,
) -> DescriptorConflictPruneResult:
    sampled = _as_long_vector(sampled_idx, name="sampled_idx")
    feats = _as_feature_matrix(features, rows=int(sampled.numel()))
    scores = compute_descriptor_conflict_scores(
        sampled_idx=sampled,
        features=feats,
        xyz=xyz,
        source_anchor_idx=source_anchor_idx,
        min_cosine=float(min_cosine),
        min_spatial_distance_m=float(min_spatial_distance_m),
        chunk_size=int(chunk_size),
        source_chunk_size=int(source_chunk_size),
        device=device,
    )

    protect_selected = torch.zeros((int(sampled.numel()),), dtype=torch.float32)
    if protect_scores is not None:
        protect = torch.as_tensor(protect_scores, dtype=torch.float32).reshape(-1).cpu()
        max_gid = int(sampled.max().item())
        if int(protect.numel()) <= max_gid:
            raise ValueError(f"protect_scores length {protect.numel()} does not cover sampled gid {max_gid}")
        protect_selected = protect[sampled].to(torch.float32)
    raw_conflicted_mask = (scores.conflict_score > 0.0) & ~scores.source_mask
    protected_conflicted_mask = (
        raw_conflicted_mask & (protect_selected >= float(min_protect_score))
        if protect_scores is not None
        else torch.zeros_like(raw_conflicted_mask)
    )
    conflicted = torch.where(raw_conflicted_mask & ~protected_conflicted_mask)[0]
    prune_limit = int(conflicted.numel())
    fraction = min(1.0, max(0.0, float(max_prune_fraction)))
    prune_limit = min(prune_limit, int(round(float(conflicted.numel()) * fraction)))
    if int(max_prune_count) > 0:
        prune_limit = min(prune_limit, int(max_prune_count))
    if int(min_keep_count) > 0:
        prune_limit = min(prune_limit, max(0, int(sampled.numel()) - int(min_keep_count)))

    keep = torch.ones((int(sampled.numel()),), dtype=torch.bool)
    pruned_positions = torch.empty((0,), dtype=torch.long)
    if prune_limit > 0:
        conflict_order = torch.argsort(scores.conflict_score[conflicted], descending=True)
        pruned_positions = conflicted[conflict_order[:prune_limit]]
        keep[pruned_positions] = False

    pruned_idx = sampled[keep].contiguous()
    pruned_feats = feats[keep].contiguous()
    metadata = {
        "method": "descriptor_conflict_pruning",
        "input_sampled_count": int(sampled.numel()),
        "output_sampled_count": int(pruned_idx.numel()),
        "source_anchor_present_count": int(scores.source_mask.sum().item()),
        "source_kept_count": int(scores.source_mask[keep].sum().item()),
        "descriptor_conflict_candidate_count": int(conflicted.numel()),
        "descriptor_conflict_pruned_count": int(pruned_positions.numel()),
        "raw_descriptor_conflict_count": int(raw_conflicted_mask.sum().item()),
        "solver_protected_conflict_count": int(protected_conflicted_mask.sum().item()),
        "min_protect_score": float(min_protect_score),
        "min_cosine": float(min_cosine),
        "min_spatial_distance_m": float(min_spatial_distance_m),
        "max_prune_fraction": float(max_prune_fraction),
        "max_prune_count": int(max_prune_count),
        "min_keep_count": int(min_keep_count),
        "max_conflict_score": float(scores.conflict_score.max().item()) if scores.conflict_score.numel() else 0.0,
        "mean_pruned_conflict_score": float(scores.conflict_score[pruned_positions].mean().item())
        if pruned_positions.numel()
        else 0.0,
    }
    return DescriptorConflictPruneResult(
        pruned_sampled_idx=pruned_idx,
        pruned_features=pruned_feats,
        keep_mask=keep,
        scores=scores,
        metadata=metadata,
    )


def _pairwise_descriptor_conflict_edges(
    *,
    sampled_idx: torch.Tensor,
    features: torch.Tensor,
    xyz: torch.Tensor,
    top_k: int,
    min_cosine: float,
    min_spatial_distance_m: float,
    chunk_size: int,
    device: str | torch.device | None,
) -> tuple[list[PairwiseDescriptorConflictEdge], DescriptorConflictScores]:
    sampled = _as_long_vector(sampled_idx, name="sampled_idx")
    feats = _as_feature_matrix(features, rows=int(sampled.numel()))
    max_gid = int(sampled.max().item())
    points = _as_xyz(xyz, min_size=max_gid + 1)
    if int(sampled.numel()) <= 1 or int(top_k) <= 0:
        empty_scores = DescriptorConflictScores(
            nearest_source_cosine=torch.ones(int(sampled.numel()), dtype=torch.float32),
            nearest_source_distance_m=torch.zeros(int(sampled.numel()), dtype=torch.float32),
            nearest_source_gid=sampled.clone(),
            conflict_score=torch.zeros(int(sampled.numel()), dtype=torch.float32),
            source_mask=torch.zeros(int(sampled.numel()), dtype=torch.bool),
        )
        return [], empty_scores

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = max(1, int(chunk_size))
    k = min(max(1, int(top_k)), int(sampled.numel()) - 1)
    feature_device = F.normalize(feats.to(dev), dim=1)
    point_device = points[sampled].to(dev)
    edges_by_pair: dict[tuple[int, int], PairwiseDescriptorConflictEdge] = {}
    conflict_score = torch.zeros(int(sampled.numel()), dtype=torch.float32)
    nearest_cosine = torch.full((int(sampled.numel()),), -float("inf"), dtype=torch.float32)
    nearest_distance = torch.zeros(int(sampled.numel()), dtype=torch.float32)
    nearest_gid = sampled.clone()

    for start in range(0, int(sampled.numel()), chunk):
        end = min(start + chunk, int(sampled.numel()))
        sims = feature_device[start:end] @ feature_device.T
        diag_cols = torch.arange(start, end, device=dev)
        sims[torch.arange(end - start, device=dev), diag_cols] = -float("inf")
        values, positions = torch.topk(sims, k=k, dim=1, largest=True)
        for row in range(end - start):
            src_pos = int(start + row)
            src_gid = int(sampled[src_pos].item())
            for col in range(k):
                dst_pos = int(positions[row, col].item())
                cosine = float(values[row, col].item())
                if cosine < float(min_cosine):
                    continue
                distance = float((point_device[src_pos] - point_device[dst_pos]).norm().item())
                if distance < float(min_spatial_distance_m):
                    continue
                spatial_factor = min(8.0, max(1.0, distance / max(float(min_spatial_distance_m), 1.0e-6)))
                score = float(cosine * spatial_factor)
                pair = (min(src_pos, dst_pos), max(src_pos, dst_pos))
                current = edges_by_pair.get(pair)
                if current is None or score > current.score:
                    edges_by_pair[pair] = PairwiseDescriptorConflictEdge(
                        src_pos=pair[0],
                        dst_pos=pair[1],
                        src_gid=int(sampled[pair[0]].item()),
                        dst_gid=int(sampled[pair[1]].item()),
                        cosine=cosine,
                        distance_m=distance,
                        score=score,
                    )
                if score > float(conflict_score[src_pos].item()):
                    conflict_score[src_pos] = float(score)
                    nearest_cosine[src_pos] = float(cosine)
                    nearest_distance[src_pos] = float(distance)
                    nearest_gid[src_pos] = int(sampled[dst_pos].item())

    for edge in edges_by_pair.values():
        for pos, other_gid in ((edge.src_pos, edge.dst_gid), (edge.dst_pos, edge.src_gid)):
            if edge.score > float(conflict_score[pos].item()):
                conflict_score[pos] = float(edge.score)
                nearest_cosine[pos] = float(edge.cosine)
                nearest_distance[pos] = float(edge.distance_m)
                nearest_gid[pos] = int(other_gid)
    nearest_cosine = torch.where(torch.isfinite(nearest_cosine), nearest_cosine, torch.ones_like(nearest_cosine))
    scores = DescriptorConflictScores(
        nearest_source_cosine=nearest_cosine.cpu(),
        nearest_source_distance_m=nearest_distance.cpu(),
        nearest_source_gid=nearest_gid.cpu(),
        conflict_score=conflict_score.cpu(),
        source_mask=torch.zeros(int(sampled.numel()), dtype=torch.bool),
    )
    return sorted(edges_by_pair.values(), key=lambda edge: edge.score, reverse=True), scores


def prune_pairwise_descriptor_conflicts(
    *,
    sampled_idx: torch.Tensor,
    features: torch.Tensor,
    xyz: torch.Tensor,
    protect_scores: torch.Tensor | None = None,
    min_protect_score: float = 0.0,
    min_cosine: float = 0.95,
    min_spatial_distance_m: float = 1.0,
    max_prune_fraction: float = 1.0,
    max_prune_count: int = 0,
    min_keep_count: int = 0,
    top_k: int = 1,
    chunk_size: int = 2048,
    device: str | torch.device | None = None,
) -> DescriptorConflictPruneResult:
    sampled = _as_long_vector(sampled_idx, name="sampled_idx")
    feats = _as_feature_matrix(features, rows=int(sampled.numel()))
    points = _as_xyz(xyz, min_size=int(sampled.max().item()) + 1)
    edges, scores = _pairwise_descriptor_conflict_edges(
        sampled_idx=sampled,
        features=feats,
        xyz=points,
        top_k=int(top_k),
        min_cosine=float(min_cosine),
        min_spatial_distance_m=float(min_spatial_distance_m),
        chunk_size=int(chunk_size),
        device=device,
    )

    protect_selected = torch.zeros((int(sampled.numel()),), dtype=torch.float32)
    if protect_scores is not None:
        protect = torch.as_tensor(protect_scores, dtype=torch.float32).reshape(-1).cpu()
        max_gid = int(sampled.max().item())
        if int(protect.numel()) <= max_gid:
            raise ValueError(f"protect_scores length {protect.numel()} does not cover sampled gid {max_gid}")
        protect_selected = protect[sampled].to(torch.float32)

    fraction = min(1.0, max(0.0, float(max_prune_fraction)))
    prune_limit = int(round(float(len(edges)) * fraction))
    if int(max_prune_count) > 0:
        prune_limit = min(prune_limit, int(max_prune_count))
    if int(min_keep_count) > 0:
        prune_limit = min(prune_limit, max(0, int(sampled.numel()) - int(min_keep_count)))

    conflict_total = torch.zeros((int(sampled.numel()),), dtype=torch.float32)
    for edge in edges:
        conflict_total[edge.src_pos] += float(edge.score)
        conflict_total[edge.dst_pos] += float(edge.score)

    keep = torch.ones((int(sampled.numel()),), dtype=torch.bool)
    pruned_positions: list[int] = []
    protected_conflict_count = 0

    def choose_prune_pos(edge: PairwiseDescriptorConflictEdge) -> int | None:
        src_protect = float(protect_selected[edge.src_pos].item())
        dst_protect = float(protect_selected[edge.dst_pos].item())
        src_is_protected = protect_scores is not None and src_protect >= float(min_protect_score)
        dst_is_protected = protect_scores is not None and dst_protect >= float(min_protect_score)
        if src_is_protected and dst_is_protected:
            return None
        if src_is_protected:
            return int(edge.dst_pos)
        if dst_is_protected:
            return int(edge.src_pos)
        src_key = (src_protect, -float(conflict_total[edge.src_pos].item()), -int(edge.src_gid))
        dst_key = (dst_protect, -float(conflict_total[edge.dst_pos].item()), -int(edge.dst_gid))
        return int(edge.src_pos if src_key < dst_key else edge.dst_pos)

    for edge in edges:
        if len(pruned_positions) >= prune_limit:
            break
        if int(min_keep_count) > 0 and int(keep.sum().item()) <= int(min_keep_count):
            break
        if not bool(keep[edge.src_pos].item()) or not bool(keep[edge.dst_pos].item()):
            continue
        prune_pos = choose_prune_pos(edge)
        if prune_pos is None:
            protected_conflict_count += 1
            continue
        keep[prune_pos] = False
        pruned_positions.append(int(prune_pos))

    pruned_idx = sampled[keep].contiguous()
    pruned_feats = feats[keep].contiguous()
    pruned_tensor = torch.tensor(pruned_positions, dtype=torch.long)
    metadata = {
        "method": "pairwise_descriptor_conflict_pruning",
        "input_sampled_count": int(sampled.numel()),
        "output_sampled_count": int(pruned_idx.numel()),
        "descriptor_conflict_edge_count": int(len(edges)),
        "descriptor_conflict_candidate_count": int(len({pos for edge in edges for pos in (edge.src_pos, edge.dst_pos)})),
        "descriptor_conflict_pruned_count": int(len(pruned_positions)),
        "solver_protected_conflict_count": int(protected_conflict_count),
        "min_protect_score": float(min_protect_score),
        "min_cosine": float(min_cosine),
        "min_spatial_distance_m": float(min_spatial_distance_m),
        "max_prune_fraction": float(max_prune_fraction),
        "max_prune_count": int(max_prune_count),
        "min_keep_count": int(min_keep_count),
        "pairwise_top_k": int(top_k),
        "max_conflict_score": float(scores.conflict_score.max().item()) if scores.conflict_score.numel() else 0.0,
        "mean_pruned_conflict_score": float(scores.conflict_score[pruned_tensor].mean().item())
        if pruned_tensor.numel()
        else 0.0,
    }
    return DescriptorConflictPruneResult(
        pruned_sampled_idx=pruned_idx,
        pruned_features=pruned_feats,
        keep_mask=keep,
        scores=scores,
        metadata=metadata,
    )
