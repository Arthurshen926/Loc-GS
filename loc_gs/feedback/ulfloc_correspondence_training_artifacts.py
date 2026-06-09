from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import torch

from loc_gs.feedback.correspondence_supervision import export_supervision_targets
from loc_gs.stdloc_native.detector_target_refinement import build_lsf_detector_target


SCHEMA_VERSION = "ulfloc_correspondence_training_artifacts_v1"


def _split_name(payload: Mapping[str, Any]) -> str:
    split = str(payload.get("split_name", payload.get("split", ""))).strip()
    if not split:
        raise ValueError("split_name is required for ULF-Loc correspondence training artifacts")
    if split.lower() == "test":
        raise ValueError("refusing to build ULF-Loc correspondence training artifacts from test split")
    return split


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _xy_to_yx(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not torch.isfinite(torch.tensor([x, y], dtype=torch.float32)).all():
        return None
    return y, x


def _normalized_xy(row: Mapping[str, Any], *, height: int, width: int) -> tuple[float, float] | None:
    value = row.get("query_xy_norm")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(x) and math.isfinite(y):
            return max(0.0, min(0.999999, x)), max(0.0, min(0.999999, y))
    yx = _xy_to_yx(row.get("keypoint_xy"))
    if yx is None:
        return None
    y, x = yx
    return (
        max(0.0, min(0.999999, x / max(1.0, float(width)))),
        max(0.0, min(0.999999, y / max(1.0, float(height)))),
    )


def _query_geometry_aware_weights(
    rows: list[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    grid_size: int = 4,
    reprojection_scale_px: float = 5.0,
) -> list[float]:
    if not rows:
        return []

    bases = [max(0.0, _float(row.get("supervision_weight"), 0.0)) for row in rows]
    spatial_bins: list[tuple[int, int]] = []
    depth_bins: list[int] = []
    depths = [max(0.0, _float(row.get("depth_m"), 0.0)) for row in rows]
    valid_depths = [value for value in depths if value > 0.0 and math.isfinite(value)]
    min_depth = min(valid_depths) if valid_depths else 0.0
    max_depth = max(valid_depths) if valid_depths else 0.0
    depth_span = max_depth - min_depth
    bins = max(1, int(grid_size))

    for row, depth in zip(rows, depths):
        xy = _normalized_xy(row, height=int(height), width=int(width))
        if xy is None:
            spatial_bins.append((-1, -1))
        else:
            x, y = xy
            spatial_bins.append((min(bins - 1, int(x * bins)), min(bins - 1, int(y * bins))))
        if depth_span > 1e-6 and depth > 0.0 and math.isfinite(depth):
            depth_bins.append(min(bins - 1, int(((depth - min_depth) / depth_span) * bins)))
        else:
            depth_bins.append(0)

    spatial_counts: dict[tuple[int, int], int] = defaultdict(int)
    depth_counts: dict[int, int] = defaultdict(int)
    for cell in spatial_bins:
        spatial_counts[cell] += 1
    for depth_bin in depth_bins:
        depth_counts[depth_bin] += 1
    spatial_mean = len(rows) / max(1, len(spatial_counts))
    depth_mean = len(rows) / max(1, len(depth_counts))

    raw: list[float] = []
    for row, base, cell, depth_bin in zip(rows, bases, spatial_bins, depth_bins):
        spatial_balance = math.sqrt(spatial_mean / max(1.0, float(spatial_counts[cell])))
        depth_balance = math.sqrt(depth_mean / max(1.0, float(depth_counts[depth_bin])))
        spatial_balance = max(0.5, min(1.5, spatial_balance))
        depth_balance = max(0.5, min(1.5, depth_balance))

        reproj = max(0.0, _float(row.get("reprojection_error_px"), 999.0))
        reproj_quality = 1.0 / (1.0 + reproj / max(1e-6, float(reprojection_scale_px)))
        margin = max(0.0, _float(row.get("descriptor_margin"), 0.0))
        margin_quality = 0.75 + min(0.5, margin)
        local_geometry = max(0.0, min(1.0, _float(row.get("local_geometry_score"), 0.0)))
        local_quality = 0.5 + 0.5 * local_geometry
        raw.append(
            float(base)
            * float(spatial_balance)
            * float(depth_balance)
            * float(reproj_quality)
            * float(margin_quality)
            * float(local_quality)
        )

    base_mean = sum(bases) / max(1, len(bases))
    raw_mean = sum(raw) / max(1, len(raw))
    if raw_mean <= 1e-12 or base_mean <= 0.0:
        return bases
    scale = base_mean / raw_mean
    weighted = []
    for base, value in zip(bases, raw):
        lower = max(0.0, 0.1 * base)
        upper = max(lower, 3.0 * max(base, 1e-6))
        weighted.append(max(lower, min(upper, value * scale)))
    return weighted


def _solver_validity_weight(row: Mapping[str, Any]) -> float:
    explicit = row.get("solver_validity")
    if explicit is not None:
        return max(0.0, min(1.0, _float(explicit, 0.0)))
    if int(row.get("label", 0)) != 1:
        return 0.0
    reproj = max(0.0, _float(row.get("reprojection_error_px"), 999.0))
    reproj_quality = 1.0 / (1.0 + reproj / 4.0)
    margin = max(0.0, _float(row.get("descriptor_margin"), 0.0))
    margin_quality = min(1.0, margin / 0.25) if margin > 0.0 else 0.0
    local_geometry = max(0.0, min(1.0, _float(row.get("local_geometry_score"), 0.0)))
    inlier_bonus = 1.0 if bool(row.get("pnp_inlier", True)) else 0.5
    validity = float(reproj_quality) * (0.5 + 0.5 * float(margin_quality)) * (0.5 + 0.5 * local_geometry) * inlier_bonus
    return max(0.0, min(1.0, float(validity)))


def _hard_negative_detector_weight(
    row: Mapping[str, Any],
    *,
    reprojection_scale_px: float = 8.0,
) -> float:
    if int(row.get("label", 0)) != 0:
        return 0.0
    if bool(row.get("pnp_inlier", False)):
        return 0.0
    base = max(0.05, max(0.0, _float(row.get("supervision_weight"), 0.0)))
    descriptor_score = max(0.0, min(1.0, _float(row.get("descriptor_score"), 0.0)))
    descriptor_margin = max(0.0, _float(row.get("descriptor_margin"), 0.0))
    low_margin_risk = 1.0 / (1.0 + descriptor_margin / 0.15)
    reproj = max(0.0, _float(row.get("reprojection_error_px"), 0.0))
    reproj_risk = min(1.0, reproj / max(1e-6, float(reprojection_scale_px)))
    local_geometry = max(0.0, min(1.0, _float(row.get("local_geometry_score"), 0.0)))
    geometry_risk = 1.0 - local_geometry
    weight = (
        float(base)
        * (0.5 + descriptor_score)
        * (0.75 + 0.50 * low_margin_risk)
        * (0.75 + 0.50 * reproj_risk)
        * (0.75 + 0.50 * geometry_risk)
    )
    return max(0.0, min(3.0, float(weight)))


def _select_detector_hard_negatives(
    candidates: list[tuple[tuple[float, float], Mapping[str, Any], int, float]],
    positives: list[tuple[tuple[float, float], Mapping[str, Any], int]],
    *,
    height: int,
    width: int,
    max_per_query: int,
    min_positive_distance_px: float,
    grid_size: int,
    max_per_cell: int,
) -> list[tuple[tuple[float, float], Mapping[str, Any], int, float]]:
    if not candidates:
        return []
    if int(max_per_query) <= 0:
        return []
    positive_yx = torch.tensor([item[0] for item in positives], dtype=torch.float32).reshape(-1, 2)
    cells: dict[tuple[int, int], int] = defaultdict(int)
    selected: list[tuple[tuple[float, float], Mapping[str, Any], int, float]] = []
    max_count = int(max_per_query)
    bins = max(1, int(grid_size))
    cell_cap = int(max_per_cell)
    for item in sorted(candidates, key=lambda value: float(value[3]), reverse=True):
        yx = torch.tensor(item[0], dtype=torch.float32).reshape(1, 2)
        if positive_yx.numel() > 0 and float(min_positive_distance_px) > 0.0:
            min_distance = torch.cdist(yx, positive_yx).min().item()
            if min_distance < float(min_positive_distance_px):
                continue
        y, x = item[0]
        cell = (
            min(bins - 1, max(0, int((float(y) / max(1.0, float(height))) * bins))),
            min(bins - 1, max(0, int((float(x) / max(1.0, float(width))) * bins))),
        )
        if cell_cap > 0 and cells[cell] >= cell_cap:
            continue
        selected.append(item)
        cells[cell] += 1
        if len(selected) >= max_count:
            break
    return selected


def _build_detector_targets(
    records: list[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    sigma_px: float,
    hard_negative_max_per_query: int = 64,
    hard_negative_min_positive_distance_px: float = 16.0,
    hard_negative_grid_size: int = 8,
    hard_negative_max_per_cell: int = 2,
    solver_validity_power: float = 0.0,
    detector_residual_alpha: float = 0.0,
) -> tuple[dict[str, dict[str, Any]], int, int, int]:
    positives_by_query: dict[str, list[tuple[tuple[float, float], Mapping[str, Any], int]]] = defaultdict(list)
    negatives_by_query: dict[str, list[tuple[tuple[float, float], Mapping[str, Any], int, float]]] = defaultdict(list)
    for row in records:
        yx = _xy_to_yx(row.get("keypoint_xy"))
        if yx is None:
            continue
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue
        try:
            gid = int(row.get("gaussian_id"))
        except (TypeError, ValueError):
            gid = -1
        if int(row.get("label", 0)) == 1:
            positives_by_query[query_id].append((yx, row, gid))
            continue
        negative_weight = _hard_negative_detector_weight(row)
        if negative_weight > 0.0:
            negatives_by_query[query_id].append((yx, row, gid, negative_weight))

    detector_targets: dict[str, dict[str, Any]] = {}
    point_count = 0
    negative_point_count = 0
    negative_candidate_count = sum(len(items) for items in negatives_by_query.values())
    for query_id in sorted(set(positives_by_query) | set(negatives_by_query)):
        items = positives_by_query.get(query_id, [])
        negative_items = _select_detector_hard_negatives(
            negatives_by_query.get(query_id, []),
            items,
            height=int(height),
            width=int(width),
            max_per_query=int(hard_negative_max_per_query),
            min_positive_distance_px=float(hard_negative_min_positive_distance_px),
            grid_size=int(hard_negative_grid_size),
            max_per_cell=int(hard_negative_max_per_cell),
        )
        yx = torch.tensor([item[0] for item in items], dtype=torch.float32).reshape(-1, 2)
        row_items = [item[1] for item in items]
        weights = torch.tensor(
            _query_geometry_aware_weights(row_items, height=int(height), width=int(width)),
            dtype=torch.float32,
        ).reshape(-1)
        solver_validity = torch.tensor(
            [_solver_validity_weight(row) for row in row_items],
            dtype=torch.float32,
        ).reshape(-1)
        residual_alpha = max(0.0, min(0.1, float(detector_residual_alpha)))
        teacher_weights = weights.clone()
        if residual_alpha > 0.0 and int(weights.numel()) > 0:
            residual_scale = (1.0 + residual_alpha * (2.0 * solver_validity - 1.0)).clamp(
                1.0 - residual_alpha,
                1.0 + residual_alpha,
            )
            weights = (weights * residual_scale).clamp_min(0.0)
        gids = torch.tensor([item[2] for item in items], dtype=torch.long)
        negative_yx = torch.tensor([item[0] for item in negative_items], dtype=torch.float32).reshape(-1, 2)
        negative_weights = torch.tensor([item[3] for item in negative_items], dtype=torch.float32).reshape(-1)
        negative_gids = torch.tensor([item[2] for item in negative_items], dtype=torch.long)
        heatmap, weight_map, target_metadata = build_lsf_detector_target(
            projected_yx=yx,
            support_weights=weights,
            solver_validity_weights=solver_validity,
            solver_validity_power=float(solver_validity_power),
            height=int(height),
            width=int(width),
            sigma_px=float(sigma_px),
        )
        detector_targets[query_id] = {
            "heatmap": heatmap[0, 0].contiguous(),
            "weight": weight_map[0, 0].contiguous(),
            "gaussian_ids": gids,
            "keypoint_yx": yx,
            "support_weights": weights,
            "teacher_support_weights": teacher_weights,
            "solver_validity_weights": solver_validity,
            "negative_gaussian_ids": negative_gids,
            "negative_keypoint_yx": negative_yx,
            "negative_weights": negative_weights,
            "positive_count": int(len(items)),
            "negative_count": int(len(negative_items)),
            "target_metadata": target_metadata,
            "detector_residual_alpha": float(residual_alpha),
        }
        point_count += int(len(items))
        negative_point_count += int(len(negative_items))
    return detector_targets, int(point_count), int(negative_point_count), int(negative_candidate_count)


def _build_match_scorer(records: list[Mapping[str, Any]], split: str) -> dict[str, Any]:
    labels = torch.tensor([int(row.get("label", 0)) for row in records], dtype=torch.long)
    weights = torch.tensor([max(0.0, _float(row.get("supervision_weight"), 0.0)) for row in records], dtype=torch.float32)
    gaussian_ids = torch.tensor([int(row.get("gaussian_id", -1)) for row in records], dtype=torch.long)
    descriptor_score = torch.tensor([_float(row.get("descriptor_score"), 0.0) for row in records], dtype=torch.float32)
    descriptor_margin = torch.tensor([_float(row.get("descriptor_margin"), 0.0) for row in records], dtype=torch.float32)
    reprojection_error = torch.tensor([_float(row.get("reprojection_error_px"), 999.0) for row in records], dtype=torch.float32)
    local_geometry = torch.tensor([_float(row.get("local_geometry_score"), 0.0) for row in records], dtype=torch.float32)
    detector_score = torch.tensor([_float(row.get("detector_score"), 0.0) for row in records], dtype=torch.float32)
    query_regression_delta = torch.tensor(
        [_float(row.get("query_regression_delta_cm"), 0.0) for row in records],
        dtype=torch.float32,
    )
    hard_negative = torch.tensor([bool(row.get("hard_negative", False)) for row in records], dtype=torch.bool)
    pose_success = torch.tensor([bool(row.get("pose_success", False)) for row in records], dtype=torch.bool)
    depth_bin = torch.tensor([int(_float(row.get("depth_bin"), -1.0)) for row in records], dtype=torch.long)
    image_cell = []
    query_xy_norm = []
    keypoint_yx = []
    for row in records:
        yx = _xy_to_yx(row.get("keypoint_xy"))
        keypoint_yx.append((float("nan"), float("nan")) if yx is None else yx)
        xy = row.get("query_xy_norm")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            query_xy_norm.append((_float(xy[0], float("nan")), _float(xy[1], float("nan"))))
        else:
            query_xy_norm.append((float("nan"), float("nan")))
        cell = row.get("image_cell")
        if isinstance(cell, (list, tuple)) and len(cell) >= 2:
            image_cell.append((int(_float(cell[0], -1.0)), int(_float(cell[1], -1.0))))
        else:
            image_cell.append((-1, -1))
    return {
        "schema_version": "ulfloc_match_scorer_targets_v1",
        "split_name": split,
        "labels": labels,
        "weights": weights,
        "gaussian_ids": gaussian_ids,
        "query_ids": [str(row.get("query_id", "")) for row in records],
        "keypoint_yx": torch.tensor(keypoint_yx, dtype=torch.float32),
        "query_xy_norm": torch.tensor(query_xy_norm, dtype=torch.float32),
        "image_cell": torch.tensor(image_cell, dtype=torch.long),
        "depth_bin": depth_bin,
        "descriptor_score": descriptor_score,
        "descriptor_margin": descriptor_margin,
        "detector_score": detector_score,
        "reprojection_error_px": reprojection_error,
        "local_geometry_score": local_geometry,
        "query_regression_delta_cm": query_regression_delta,
        "hard_negative": hard_negative,
        "pose_success": pose_success,
        "source_role": [str(row.get("source_role", "")) for row in records],
    }


def build_ulfloc_correspondence_training_artifacts(
    correspondence_supervision: Mapping[str, Any],
    *,
    height: int,
    width: int,
    sigma_px: float = 1.0,
    hard_negative_max_per_query: int = 64,
    hard_negative_min_positive_distance_px: float = 16.0,
    hard_negative_grid_size: int = 8,
    hard_negative_max_per_cell: int = 2,
    solver_validity_power: float = 0.0,
    detector_residual_alpha: float = 0.0,
) -> dict[str, Any]:
    split = _split_name(correspondence_supervision)
    rows = list(correspondence_supervision.get("records", []))
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("correspondence_supervision records must be mappings")
    records: list[Mapping[str, Any]] = [row for row in rows if isinstance(row, Mapping)]
    (
        detector_targets,
        positive_detector_points,
        hard_negative_detector_points,
        hard_negative_detector_candidates,
    ) = _build_detector_targets(
        records,
        height=int(height),
        width=int(width),
        sigma_px=float(sigma_px),
        hard_negative_max_per_query=int(hard_negative_max_per_query),
        hard_negative_min_positive_distance_px=float(hard_negative_min_positive_distance_px),
        hard_negative_grid_size=int(hard_negative_grid_size),
        hard_negative_max_per_cell=int(hard_negative_max_per_cell),
        solver_validity_power=float(solver_validity_power),
        detector_residual_alpha=float(detector_residual_alpha),
    )
    exported_targets = export_supervision_targets({"records": records})
    match_scorer = _build_match_scorer(records, split)
    ranking = dict(exported_targets["pnp_ranking"])
    conflict_graph = dict(exported_targets["conflict_graph"])
    hard_negative_multipliers = [
        max(1.0, _float(row.get("negative_supervision_multiplier"), 1.0))
        for row in records
        if bool(row.get("hard_negative", False))
    ]
    positive_solver_validity = [
        _solver_validity_weight(row)
        for row in records
        if int(row.get("label", 0)) == 1
    ]
    split_audit = {
        "schema_version": "ulfloc_correspondence_training_split_audit_v1",
        "split_name": split,
        "test_split_used": False,
        "audit_status": "passed",
    }
    metrics = {
        "schema_version": "ulfloc_correspondence_training_metrics_v1",
        "detector_weighting": "pnp_geometry_aware_selective_hard_negative_suppression_v3",
        "detector_solver_validity_power": float(max(0.0, solver_validity_power)),
        "detector_residual_alpha": float(max(0.0, min(0.1, detector_residual_alpha))),
        "detector_residual_mode": "teacher_residual_clamped" if float(detector_residual_alpha) > 0.0 else "off",
        "record_count": int(len(records)),
        "positive_count": int(sum(int(row.get("label", 0)) for row in records)),
        "negative_count": int(sum(1 for row in records if int(row.get("label", 0)) == 0)),
        "hard_negative_record_count": int(len(hard_negative_multipliers)),
        "mean_hard_negative_multiplier": float(
            sum(hard_negative_multipliers) / max(1, len(hard_negative_multipliers))
        )
        if hard_negative_multipliers
        else 0.0,
        "mean_positive_solver_validity": float(sum(positive_solver_validity) / max(1, len(positive_solver_validity)))
        if positive_solver_validity
        else 0.0,
        "detector_query_count": int(len(detector_targets)),
        "positive_detector_point_count": int(positive_detector_points),
        "hard_negative_detector_point_count": int(hard_negative_detector_points),
        "hard_negative_detector_candidate_count": int(hard_negative_detector_candidates),
        "hard_negative_max_per_query": int(hard_negative_max_per_query),
        "hard_negative_min_positive_distance_px": float(hard_negative_min_positive_distance_px),
        "hard_negative_grid_size": int(hard_negative_grid_size),
        "hard_negative_max_per_cell": int(hard_negative_max_per_cell),
        "match_scorer_row_count": int(match_scorer["labels"].numel()),
        "ranking_preference_count": int(len(ranking.get("pairwise_preferences", []))),
        "conflict_edge_count": int(len(conflict_graph.get("edges", []))),
        "height": int(height),
        "width": int(width),
        "sigma_px": float(sigma_px),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "split_name": split,
        "detector_targets": detector_targets,
        "match_scorer": match_scorer,
        "pnp_ranking": ranking,
        "conflict_graph": conflict_graph,
        "metadata": metrics,
        "split_audit": split_audit,
    }
