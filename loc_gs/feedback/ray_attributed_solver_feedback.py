from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch


SCHEMA_VERSION = "ray_attributed_solver_feedback_v1"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _gaussian_id(item: Mapping[str, Any]) -> int:
    value = item.get("gaussian_id", item.get("matched_gaussian_id"))
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"contributor has invalid gaussian_id: {value!r}") from exc
    if out < 0:
        raise ValueError(f"contributor has negative gaussian_id: {out}")
    return out


def validate_split_name(split_name: str) -> str:
    normalized = str(split_name).strip()
    if not normalized:
        raise ValueError("split_name is required")
    if normalized.lower() == "test":
        raise ValueError("test split is not allowed for ray-attributed solver feedback")
    return normalized


def normalize_contributors(
    contributors: Sequence[Mapping[str, Any]],
    *,
    min_weight: float = 1e-8,
) -> list[dict[str, float | int]]:
    """Normalize ray contributors using alpha/contribution and reliability."""

    weighted: list[dict[str, float | int]] = []
    total = 0.0
    for item in contributors:
        contribution = _float(item.get("contribution", item.get("alpha", item.get("weight", 0.0))))
        reliability = _float(item.get("reliability", item.get("visibility", 1.0)), 1.0)
        weight = max(0.0, contribution) * max(0.0, reliability)
        if weight <= 0.0:
            continue
        row = {"gaussian_id": _gaussian_id(item), "raw_weight": float(weight), "weight": 0.0}
        weighted.append(row)
        total += weight

    if total <= 0.0:
        return []

    normalized = []
    for row in weighted:
        weight = float(row["raw_weight"]) / total
        if weight >= float(min_weight):
            normalized.append(
                {
                    "gaussian_id": int(row["gaussian_id"]),
                    "raw_weight": float(row["raw_weight"]),
                    "weight": float(weight),
                }
            )
    return normalized


def soft_point_from_contributors(contributors: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    normalized = normalize_contributors(contributors)
    if not normalized:
        raise ValueError("at least one positive contributor is required")

    xyz_by_gid: dict[int, torch.Tensor] = {}
    for item in contributors:
        gid = _gaussian_id(item)
        if "xyz" not in item:
            continue
        xyz = torch.as_tensor(item["xyz"], dtype=torch.float32).reshape(-1)
        if xyz.numel() != 3:
            raise ValueError(f"contributor xyz must have 3 values, got {xyz.numel()}")
        xyz_by_gid[gid] = xyz

    if not xyz_by_gid:
        raise ValueError("contributors must include xyz to compute a soft point")

    point = torch.zeros(3, dtype=torch.float32)
    for item in normalized:
        gid = int(item["gaussian_id"])
        if gid not in xyz_by_gid:
            raise ValueError(f"missing xyz for contributor gaussian_id={gid}")
        point += float(item["weight"]) * xyz_by_gid[gid]
    return point


def _composition_entropy(normalized: Sequence[Mapping[str, Any]]) -> float:
    if not normalized:
        return 0.0
    entropy = 0.0
    for item in normalized:
        weight = max(float(item.get("weight", 0.0)), 1e-12)
        entropy -= weight * math.log(weight)
    return float(entropy / max(math.log(len(normalized)), 1e-12)) if len(normalized) > 1 else 0.0


def artifact_score_from_ray(
    contributors: Sequence[Mapping[str, Any]],
    *,
    expected_depth: float | None = None,
    rendered_depth: float | None = None,
    depth_margin: float = 1.0,
) -> float:
    """Estimate whether a near, high-alpha contributor is occluding the expected ray target."""

    if expected_depth is None or rendered_depth is None:
        return 0.0
    expected = _float(expected_depth)
    rendered = _float(rendered_depth)
    if expected <= 0.0 or rendered <= 0.0 or rendered >= expected - float(depth_margin):
        return 0.0

    normalized = normalize_contributors(contributors)
    if not normalized:
        return 0.0
    contributor_by_gid = {_gaussian_id(item): item for item in contributors}
    near_mass = 0.0
    for item in normalized:
        raw = contributor_by_gid.get(int(item["gaussian_id"]), {})
        depth = _float(raw.get("depth", raw.get("z", rendered)), rendered)
        if depth <= expected - float(depth_margin):
            near_mass += float(item["weight"])
    depth_gap = min(1.0, max(0.0, (expected - rendered) / max(expected, 1e-6)))
    entropy = _composition_entropy(normalized)
    depth_adjusted = near_mass * (0.5 + 0.5 * depth_gap) * (1.0 - 0.25 * entropy)
    return max(0.0, min(1.0, max(near_mass, depth_adjusted)))


def _observation_quality(record: Mapping[str, Any], reprojection_threshold_px: float) -> float:
    descriptor = max(0.0, min(1.0, _float(record.get("descriptor_score"), 1.0)))
    local_geometry = max(0.0, min(1.0, _float(record.get("local_geometry_score"), 1.0)))
    reproj = _float(record.get("reprojection_error_px"), 0.0)
    reproj_quality = max(0.0, min(1.0, 1.0 - reproj / max(float(reprojection_threshold_px), 1e-6)))
    return float(descriptor * local_geometry * reproj_quality)


def _cell_from_record(record: Mapping[str, Any]) -> list[int] | None:
    for key in ("image_cell", "cell", "query_cell", "keypoint_cell"):
        raw_cell = record.get(key)
        if isinstance(raw_cell, (list, tuple)) and len(raw_cell) >= 2:
            try:
                return [int(raw_cell[0]), int(raw_cell[1])]
            except (TypeError, ValueError):
                return None
    for x_key, y_key in (("cell_x", "cell_y"), ("grid_x", "grid_y"), ("image_cell_x", "image_cell_y")):
        if x_key in record and y_key in record:
            try:
                return [int(record[x_key]), int(record[y_key])]
            except (TypeError, ValueError):
                return None
    return None


def _depth_bin_from_record(record: Mapping[str, Any]) -> int | None:
    for key in ("depth_bin", "query_depth_bin"):
        if key in record:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                return None
    for key in ("expected_depth", "rendered_depth", "depth", "depth_m"):
        if key in record:
            value = _float(record.get(key), -1.0)
            if value > 0.0:
                return int(math.floor(value))
    return None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _vector_from_record(record: Mapping[str, Any], keys: tuple[str, ...], length: int) -> list[float] | None:
    for key in keys:
        raw = record.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= int(length):
            values: list[float] = []
            ok = True
            for index in range(int(length)):
                value = _finite_float(raw[index])
                if value is None:
                    ok = False
                    break
                values.append(float(value))
            if ok:
                return values
    return None


def _xy_norm_from_record(record: Mapping[str, Any]) -> list[float] | None:
    direct = _vector_from_record(
        record,
        ("xy_norm", "query_xy_norm", "normalized_xy", "query_normalized_xy", "uv_norm", "keypoint_xy_norm"),
        2,
    )
    if direct is not None:
        return direct
    for x_key, y_key in (
        ("x_norm", "y_norm"),
        ("query_x_norm", "query_y_norm"),
        ("u_norm", "v_norm"),
        ("normalized_x", "normalized_y"),
    ):
        if x_key not in record or y_key not in record:
            continue
        x_value = _finite_float(record.get(x_key))
        y_value = _finite_float(record.get(y_key))
        if x_value is not None and y_value is not None:
            return [float(x_value), float(y_value)]
    return None


def _scalar_from_record(record: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in record:
            continue
        value = _finite_float(record.get(key))
        if value is not None:
            return float(value)
    return None


def _query_observation_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cell = _cell_from_record(record)
    if cell is not None:
        out["cell"] = cell
    depth_bin = _depth_bin_from_record(record)
    if depth_bin is not None:
        out["depth_bin"] = int(depth_bin)
    xy_norm = _xy_norm_from_record(record)
    if xy_norm is not None:
        out["xy_norm"] = xy_norm
    bearing = _vector_from_record(
        record,
        ("bearing", "bearing_vector", "unit_bearing", "ray", "ray_direction", "query_bearing"),
        3,
    )
    if bearing is not None:
        out["bearing"] = bearing
    camera_xyz = _vector_from_record(
        record,
        ("camera_xyz", "camera_point", "point_cam", "query_camera_xyz", "xyz_cam", "camera_frame_xyz"),
        3,
    )
    if camera_xyz is not None:
        out["camera_xyz"] = camera_xyz
    depth = _scalar_from_record(record, ("expected_depth", "depth", "depth_m", "query_depth"))
    if depth is not None:
        out["depth"] = float(depth)
    for output_key, keys in (
        ("reprojection_error_px", ("reprojection_error_px", "reproj_error_px")),
        ("descriptor_margin", ("descriptor_margin", "match_margin", "nn_margin")),
        ("local_geometry_score", ("local_geometry_score", "geometry_score", "lgcv_score")),
        ("ray_entropy", ("ray_entropy", "composition_entropy", "alpha_entropy")),
    ):
        value = _scalar_from_record(record, keys)
        if value is not None:
            out[output_key] = float(value)
    contributors = record.get("contributors", [])
    normalized = normalize_contributors(contributors) if isinstance(contributors, Sequence) else []
    ray_artifact = artifact_score_from_ray(
        contributors if isinstance(contributors, Sequence) else [],
        expected_depth=record.get("expected_depth"),
        rendered_depth=record.get("rendered_depth"),
    )
    explicit_artifact = _scalar_from_record(record, ("ray_artifact_score", "artifact_score", "artifact_risk"))
    artifact = max(float(ray_artifact), float(explicit_artifact or 0.0))
    if artifact > 0.0:
        out["ray_artifact_score"] = float(max(0.0, min(1.0, artifact)))
    if "ray_entropy" not in out and normalized:
        out["ray_entropy"] = float(_composition_entropy(normalized))
    return out


def accumulate_ray_feedback(
    observations: Iterable[Mapping[str, Any]],
    *,
    num_gaussians: int,
    split_name: str,
    reprojection_threshold_px: float = 8.0,
    weak_outlier_weight: float = 0.02,
    artifact_weight: float = 1.0,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    count = int(num_gaussians)
    if count < 0:
        raise ValueError(f"num_gaussians must be non-negative, got {count}")

    observed_count = torch.zeros(count, dtype=torch.long)
    positive_observed_count = torch.zeros(count, dtype=torch.long)
    contribution_mass = torch.zeros(count, dtype=torch.float32)
    positive_contribution_mass = torch.zeros(count, dtype=torch.float32)
    weighted_support = torch.zeros(count, dtype=torch.float32)
    hard_negative = torch.zeros(count, dtype=torch.float32)
    artifact_risk = torch.zeros(count, dtype=torch.float32)
    source_views: set[str] = set()
    query_ids: set[str] = set()
    per_query_support: dict[str, dict[int, float]] = {}
    per_query_negative_support: dict[str, dict[int, float]] = {}
    per_query_observations: dict[str, dict[int, dict[str, Any]]] = {}
    per_query_observation_scores: dict[tuple[str, int], float] = {}
    observation_count = 0

    for record in observations:
        record_split = str(record.get("split_name", split)).strip() or split
        if record_split.lower() == "test":
            raise ValueError("test split is not allowed for ray-attributed solver feedback")
        contributors = record.get("contributors", [])
        if not isinstance(contributors, Sequence):
            raise TypeError("observation contributors must be a sequence")
        normalized = normalize_contributors(contributors)
        if not normalized:
            continue

        observation_count += 1
        if record.get("source_view_id"):
            source_views.add(str(record["source_view_id"]))
        if record.get("query_id"):
            query_ids.add(str(record["query_id"]))

        pnp_inlier = _bool(record.get("pnp_inlier", False))
        quality = _observation_quality(record, float(reprojection_threshold_px)) if pnp_inlier else 0.0
        explicit_artifact = max(0.0, min(1.0, _float(record.get("artifact_score"), 0.0)))
        ray_artifact = artifact_score_from_ray(
            contributors,
            expected_depth=record.get("expected_depth"),
            rendered_depth=record.get("rendered_depth"),
        )
        artifact = max(explicit_artifact, ray_artifact)
        reproj = _float(record.get("reprojection_error_px"), 0.0)
        descriptor = _float(record.get("descriptor_score"), 0.0)
        weak_risk = (
            float(weak_outlier_weight)
            if (not pnp_inlier and descriptor >= 0.5 and reproj > float(reprojection_threshold_px))
            else 0.0
        )

        for item in normalized:
            gid = int(item["gaussian_id"])
            if gid >= count:
                raise ValueError(f"gaussian_id {gid} is outside num_gaussians={count}")
            weight = float(item["weight"])
            observed_count[gid] += 1
            contribution_mass[gid] += weight
            artifact_risk[gid] += float(artifact_weight) * artifact * weight
            hard_negative[gid] += weak_risk * weight
            if pnp_inlier and quality > 0.0:
                positive_observed_count[gid] += 1
                positive_contribution_mass[gid] += weight
                support_value = float(quality * weight)
                weighted_support[gid] += support_value
                query_id = str(record.get("query_id", "")).strip()
                if query_id:
                    query_support = per_query_support.setdefault(query_id, {})
                    query_support[gid] = float(query_support.get(gid, 0.0) + support_value)
                    query_observation = _query_observation_from_record(record)
                    previous = per_query_observation_scores.get((query_id, gid), -1.0)
                    if query_observation and support_value > previous:
                        per_query_observations.setdefault(query_id, {})[gid] = query_observation
                        per_query_observation_scores[(query_id, gid)] = support_value
            negative_value = float(max(float(weak_risk), float(artifact) * float(artifact_weight)) * weight)
            query_id = str(record.get("query_id", "")).strip()
            if negative_value > 0.0 and query_id:
                query_negative = per_query_negative_support.setdefault(query_id, {})
                query_negative[gid] = float(query_negative.get(gid, 0.0) + negative_value)

    support_score = weighted_support
    artifact_risk = artifact_risk / contribution_mass.clamp_min(1e-6)
    hard_negative_risk = hard_negative / contribution_mass.clamp_min(1e-6)
    return {
        "schema_version": SCHEMA_VERSION,
        "support_score": support_score.to(torch.float32),
        "weighted_support": weighted_support.to(torch.float32),
        "contribution_mass": contribution_mass.to(torch.float32),
        "positive_contribution_mass": positive_contribution_mass.to(torch.float32),
        "observed_count": observed_count,
        "positive_observed_count": positive_observed_count,
        "hard_negative_risk": hard_negative_risk.to(torch.float32).clamp(0.0, 1.0),
        "artifact_risk": artifact_risk.to(torch.float32).clamp(0.0, 1.0),
        "per_query_support": {
            str(query_id): {int(gid): float(value) for gid, value in sorted(support.items())}
            for query_id, support in sorted(per_query_support.items())
        },
        "per_query_negative_support": {
            str(query_id): {int(gid): float(value) for gid, value in sorted(support.items())}
            for query_id, support in sorted(per_query_negative_support.items())
        },
        "per_query_observations": {
            str(query_id): {int(gid): dict(value) for gid, value in sorted(observations.items())}
            for query_id, observations in sorted(per_query_observations.items())
        },
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "split_name": split,
            "num_gaussians": int(count),
            "observation_count": int(observation_count),
            "query_count": int(len(query_ids)),
            "source_view_count": int(len(source_views)),
            "per_query_support_query_count": int(len(per_query_support)),
            "per_query_support_entry_count": int(sum(len(support) for support in per_query_support.values())),
            "per_query_negative_support_query_count": int(len(per_query_negative_support)),
            "per_query_negative_support_entry_count": int(
                sum(len(support) for support in per_query_negative_support.values())
            ),
            "per_query_observation_query_count": int(len(per_query_observations)),
            "per_query_observation_entry_count": int(sum(len(obs) for obs in per_query_observations.values())),
            "hyperparameters": {
                "reprojection_threshold_px": float(reprojection_threshold_px),
                "weak_outlier_weight": float(weak_outlier_weight),
                "artifact_weight": float(artifact_weight),
            },
        },
    }
