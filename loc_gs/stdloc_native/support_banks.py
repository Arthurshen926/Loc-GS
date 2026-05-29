from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

_FORBIDDEN_QUERY_FIELDS = {
    "gt_pose",
    "pose",
    "ground_truth",
    "translation_error",
    "rotation_error",
    "te_cm",
    "re_deg",
    "dense_te_cm",
    "sparse_te_cm",
    "eval_result",
}


def _as_feature_matrix(query_features: torch.Tensor | Any) -> torch.Tensor:
    features = torch.as_tensor(query_features, dtype=torch.float32).cpu()
    if features.dim() == 1:
        features = features.reshape(-1, 1)
    if features.dim() != 2:
        raise ValueError("query_features must have shape [Q,D]")
    return features


def _partition_by_first_axis(features: torch.Tensor, bank_count: int) -> list[list[int]]:
    count = int(features.shape[0])
    banks = max(1, min(int(bank_count), count))
    order = torch.argsort(features[:, 0], stable=True).tolist()
    partitions: list[list[int]] = []
    for bank_id in range(banks):
        start = int(round(bank_id * count / banks))
        end = int(round((bank_id + 1) * count / banks))
        partitions.append([int(idx) for idx in order[start:end]])
    return partitions


def _normalize_weights(weights: dict[int, float]) -> dict[int, float]:
    if not weights:
        return {}
    max_value = max(max(float(value), 0.0) for value in weights.values())
    if max_value <= 0.0:
        return {int(key): 0.0 for key in weights}
    return {int(key): float(value) / max_value for key, value in weights.items()}


def _reject_forbidden_query_fields(observations: Sequence[Mapping[str, Any]]) -> None:
    for row in observations:
        forbidden = sorted(_FORBIDDEN_QUERY_FIELDS & set(row.keys()))
        if forbidden:
            raise ValueError(f"forbidden query fields for paper-safe bank routing: {forbidden}")


def build_observable_query_features(query_observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build route features from query-observable signals only."""

    _reject_forbidden_query_fields(query_observations)
    query_ids: list[str] = []
    rows: list[list[float]] = []
    for index, obs in enumerate(query_observations):
        query_ids.append(str(obs.get("query_id", obs.get("image_id", f"query_{index:06d}"))))
        centroid = obs.get("detector_centroid_yx", (0.5, 0.5))
        centroid_t = torch.as_tensor(centroid, dtype=torch.float32).reshape(-1)
        if centroid_t.numel() < 2:
            raise ValueError("detector_centroid_yx must contain y and x")
        keypoint_count = float(obs.get("keypoint_count", 0.0) or 0.0)
        entropy = float(obs.get("topk_match_entropy", obs.get("match_entropy", 0.0)) or 0.0)
        sparse_inliers = float(obs.get("sparse_inliers", 0.0) or 0.0)
        confidence = float(obs.get("pnp_confidence", 0.0) or 0.0)
        global_desc = torch.as_tensor(obs.get("global_descriptor", []), dtype=torch.float32).reshape(-1)
        global_mean = float(global_desc.mean().item()) if global_desc.numel() else 0.0
        global_std = float(global_desc.std(unbiased=False).item()) if global_desc.numel() else 0.0
        rows.append(
            [
                max(0.0, keypoint_count) / 2048.0,
                float(centroid_t[0]),
                float(centroid_t[1]),
                max(0.0, entropy),
                max(0.0, sparse_inliers) / 2048.0,
                max(0.0, min(1.0, confidence)),
                global_mean,
                global_std,
            ]
        )
    features = torch.tensor(rows, dtype=torch.float32) if rows else torch.empty(0, 8)
    return {
        "query_ids": query_ids,
        "features": features,
        "feature_names": [
            "keypoint_count_norm",
            "detector_centroid_y",
            "detector_centroid_x",
            "topk_match_entropy",
            "sparse_inliers_norm",
            "pnp_confidence",
            "global_descriptor_mean",
            "global_descriptor_std",
        ],
        "metadata": {
            "feature_source": "query_observable_only",
            "forbidden_fields_rejected": sorted(_FORBIDDEN_QUERY_FIELDS),
        },
    }


def build_query_support_banks(
    *,
    query_ids: Sequence[str],
    query_features: torch.Tensor | Any,
    landmark_support: Mapping[int, Mapping[str, float]],
    bank_count: int,
    normalize_landmark_weights: bool = True,
) -> dict[str, Any]:
    """Build deterministic query-conditioned support banks from train/self-map evidence."""

    features = _as_feature_matrix(query_features)
    if len(query_ids) != int(features.shape[0]):
        raise ValueError("query_ids length must match query_features rows")
    partitions = _partition_by_first_axis(features, int(bank_count))
    banks: list[dict[str, Any]] = []
    centroids: list[torch.Tensor] = []
    for bank_id, indices in enumerate(partitions):
        bank_queries = [str(query_ids[idx]) for idx in indices]
        bank_set = set(bank_queries)
        weights: dict[int, float] = {}
        for landmark_id, per_query in landmark_support.items():
            value = sum(float(score) for query_id, score in per_query.items() if str(query_id) in bank_set)
            if value > 0.0:
                weights[int(landmark_id)] = float(value)
        if normalize_landmark_weights:
            weights = _normalize_weights(weights)
        centroid = features[indices].mean(dim=0) if indices else torch.zeros(features.shape[1], dtype=torch.float32)
        centroids.append(centroid)
        banks.append(
            {
                "bank_id": int(bank_id),
                "query_ids": bank_queries,
                "centroid": centroid.tolist(),
                "landmark_weights": weights,
            }
        )
    return {
        "banks": banks,
        "router": {
            "mode": "nearest_centroid",
            "centroids": torch.stack(centroids, dim=0) if centroids else torch.empty(0, features.shape[1]),
            "bank_ids": [bank["bank_id"] for bank in banks],
        },
        "metadata": {
            "query_count": int(features.shape[0]),
            "bank_count": int(len(banks)),
            "paper_safe_scope": "offline_train_or_selfmap_router",
        },
    }


def _support_for_queries(
    landmark_support: Mapping[int, Mapping[str, float]],
    query_ids: set[str],
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for landmark_id, per_query in landmark_support.items():
        value = sum(float(score) for query_id, score in per_query.items() if str(query_id) in query_ids)
        if value > 0.0:
            weights[int(landmark_id)] = float(value)
    return _normalize_weights(weights)


def build_named_support_banks(
    *,
    query_observations: Sequence[Mapping[str, Any]],
    landmark_support: Mapping[int, Mapping[str, float]],
    native_safe_core_ids: Sequence[int],
    hard_query_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build fixed v8 support banks from observable train/self-map query signals."""

    feature_payload = build_observable_query_features(query_observations)
    query_ids = feature_payload["query_ids"]
    observed_query_ids = {str(query_id) for query_id in query_ids}
    features = feature_payload["features"]
    id_by_index = {idx: str(query_id) for idx, query_id in enumerate(query_ids)}
    ambiguity_queries: set[str] = set()
    occlusion_queries: set[str] = set()
    for idx, obs in enumerate(query_observations):
        query_id = id_by_index[idx]
        entropy = float(obs.get("topk_match_entropy", obs.get("match_entropy", 0.0)) or 0.0)
        keypoints = float(obs.get("keypoint_count", 0.0) or 0.0)
        inliers = float(obs.get("sparse_inliers", 0.0) or 0.0)
        confidence = float(obs.get("pnp_confidence", 0.0) or 0.0)
        if entropy >= 0.65:
            ambiguity_queries.add(query_id)
        if keypoints < 80.0 or inliers < 16.0 or confidence < 0.30:
            occlusion_queries.add(query_id)
    requested_hard_queries = {str(query_id) for query_id in hard_query_ids}
    hard_queries = requested_hard_queries & observed_query_ids
    native_weights = {int(gid): 1.0 for gid in native_safe_core_ids}
    banks = [
        {
            "bank_id": 0,
            "name": "native_safe_core",
            "query_ids": query_ids,
            "landmark_weights": native_weights,
        },
        {
            "bank_id": 1,
            "name": "ambiguity_safe",
            "query_ids": sorted(ambiguity_queries),
            "landmark_weights": _support_for_queries(landmark_support, ambiguity_queries),
        },
        {
            "bank_id": 2,
            "name": "occlusion_robust",
            "query_ids": sorted(occlusion_queries),
            "landmark_weights": _support_for_queries(landmark_support, occlusion_queries),
        },
        {
            "bank_id": 3,
            "name": "hard_tail_recovery",
            "query_ids": sorted(hard_queries),
            "landmark_weights": _support_for_queries(landmark_support, hard_queries),
        },
    ]
    centroids = []
    query_id_to_idx = {str(query_id): idx for idx, query_id in enumerate(query_ids)}
    for bank in banks:
        ids = [query_id_to_idx[qid] for qid in bank["query_ids"] if qid in query_id_to_idx]
        if ids:
            centroids.append(features[ids].mean(dim=0))
        elif features.numel():
            centroids.append(features.mean(dim=0))
        else:
            centroids.append(torch.zeros(8, dtype=torch.float32))
    return {
        "banks": banks,
        "query_ids": query_ids,
        "query_features": features,
        "feature_names": feature_payload["feature_names"],
        "router": {
            "mode": "nearest_centroid",
            "centroids": torch.stack(centroids, dim=0),
            "bank_ids": [bank["bank_id"] for bank in banks],
            "bank_names": [bank["name"] for bank in banks],
            "non_empty_bank_ids": [
                bank["bank_id"]
                for bank in banks
                if len(bank.get("landmark_weights", {})) > 0
            ],
        },
        "metadata": {
            "bank_count": int(len(banks)),
            "paper_safe_scope": "offline_train_or_selfmap_router",
            "routing_uses_gt_or_eval_result": False,
            "requested_hard_query_count": int(len(requested_hard_queries)),
            "matched_hard_query_count": int(len(hard_queries)),
            "unmatched_hard_query_count": int(len(requested_hard_queries - observed_query_ids)),
        },
    }


def route_query_to_banks(query_feature: torch.Tensor | Any, router: Mapping[str, Any], *, topk: int = 1) -> list[int]:
    """Route a query to precomputed support banks using observable features only."""

    if str(router.get("mode", "")) != "nearest_centroid":
        raise ValueError("only nearest_centroid router is supported")
    centroids = _as_feature_matrix(router["centroids"])
    feature = torch.as_tensor(query_feature, dtype=torch.float32).reshape(1, -1).cpu()
    if feature.shape[1] != centroids.shape[1]:
        raise ValueError("query_feature dimension must match router centroids")
    bank_ids = [int(item) for item in router.get("bank_ids", list(range(int(centroids.shape[0]))))]
    non_empty_raw = router.get("non_empty_bank_ids")
    allowed = {int(item) for item in non_empty_raw} if non_empty_raw is not None else set(bank_ids)
    allowed_indices = [idx for idx, bank_id in enumerate(bank_ids) if int(bank_id) in allowed]
    if not allowed_indices:
        allowed_indices = list(range(len(bank_ids)))
    distance = torch.linalg.norm(centroids - feature, dim=1)
    masked = torch.full_like(distance, float("inf"))
    masked[torch.as_tensor(allowed_indices, dtype=torch.long)] = distance[torch.as_tensor(allowed_indices, dtype=torch.long)]
    order = torch.argsort(masked, stable=True)[: max(1, min(int(topk), len(allowed_indices)))].tolist()
    return [bank_ids[int(idx)] for idx in order]
