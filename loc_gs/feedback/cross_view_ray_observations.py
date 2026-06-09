from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name


SCHEMA_VERSION = "cross_view_ray_observations_v1"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _xy(record: Mapping[str, Any], *keys: str) -> list[float]:
    for key in keys:
        if key in record:
            value = record[key]
            break
    else:
        raise ValueError(f"record is missing one of coordinate fields: {keys}")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"coordinate field must be a sequence: {value!r}")
    values = list(value)
    if len(values) != 2:
        raise ValueError(f"coordinate field must contain 2 values, got {len(values)}")
    return [_float(values[0]), _float(values[1])]


def _vector(record: Mapping[str, Any], key: str, *, length: int) -> list[float] | None:
    if key not in record:
        return None
    value = record[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    values = list(value)
    if len(values) < int(length):
        return None
    return [_float(values[index]) for index in range(int(length))]


def _record_split(record: Mapping[str, Any], default: str) -> str:
    split = str(record.get("split_name", record.get("split", default))).strip() or default
    return validate_split_name(split)


def pixel_key(xy: Sequence[float], *, quantization_px: float = 1.0) -> tuple[int, int]:
    if quantization_px <= 0.0:
        raise ValueError(f"quantization_px must be positive, got {quantization_px}")
    if len(xy) != 2:
        raise ValueError(f"xy must contain 2 values, got {len(xy)}")
    return (int(round(float(xy[0]) / quantization_px)), int(round(float(xy[1]) / quantization_px)))


def build_ray_contributor_index(
    rays: Sequence[Mapping[str, Any]],
    *,
    quantization_px: float = 1.0,
    split_name: str | None = None,
) -> dict[str, Any]:
    if quantization_px <= 0.0:
        raise ValueError(f"quantization_px must be positive, got {quantization_px}")
    default_split = validate_split_name(split_name) if split_name is not None else None
    records_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for ray in rays:
        if default_split is not None:
            _record_split(ray, default_split)
        source_view_id = str(ray.get("source_view_id", ray.get("view_id", ""))).strip()
        if not source_view_id:
            raise ValueError("ray record is missing source_view_id")
        pixel_xy = _xy(ray, "pixel_xy", "source_xy", "xy")
        contributors = ray.get("contributors", [])
        if not isinstance(contributors, Sequence) or isinstance(contributors, (str, bytes)):
            raise ValueError("ray contributors must be a sequence")
        row = dict(ray)
        row["source_view_id"] = source_view_id
        row["pixel_xy"] = pixel_xy
        row["contributors"] = [dict(item) for item in contributors]
        key = (source_view_id, *pixel_key(pixel_xy, quantization_px=quantization_px))
        records_by_key.setdefault(key, []).append(row)
    return {"schema_version": SCHEMA_VERSION, "quantization_px": float(quantization_px), "records_by_key": records_by_key}


def find_ray_contributors(
    index: Mapping[str, Any],
    source_view_id: str,
    source_xy: Sequence[float],
    *,
    radius_px: float = 0.0,
    quantization_px: float | None = None,
) -> dict[str, Any] | None:
    q = float(quantization_px if quantization_px is not None else index.get("quantization_px", 1.0))
    xy = [float(source_xy[0]), float(source_xy[1])]
    source = str(source_view_id)
    records_by_key = index.get("records_by_key", {})
    base_x, base_y = pixel_key(xy, quantization_px=q)
    radius = max(0.0, float(radius_px))
    key_radius = int(math.ceil(radius / q)) if radius > 0.0 else 0

    best: dict[str, Any] | None = None
    best_distance = float("inf")
    for dx in range(-key_radius, key_radius + 1):
        for dy in range(-key_radius, key_radius + 1):
            for candidate in records_by_key.get((source, base_x + dx, base_y + dy), []):
                pixel_xy = candidate["pixel_xy"]
                distance = math.hypot(float(pixel_xy[0]) - xy[0], float(pixel_xy[1]) - xy[1])
                if radius > 0.0 and distance > radius:
                    continue
                if distance < best_distance:
                    best = candidate
                    best_distance = distance
    return dict(best) if best is not None else None


def build_cross_view_ray_observations(
    matches: Sequence[Mapping[str, Any]],
    rays: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    radius_px: float = 0.0,
    quantization_px: float = 1.0,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    index = build_ray_contributor_index(rays, quantization_px=quantization_px, split_name=split)
    observations: list[dict[str, Any]] = []
    dropped_no_contributors = 0
    query_ids: set[str] = set()
    source_view_ids: set[str] = set()

    for match in matches:
        _record_split(match, split)
        source_view_id = str(match.get("source_view_id", match.get("view_id", ""))).strip()
        if not source_view_id:
            raise ValueError("match record is missing source_view_id")
        source_xy = _xy(match, "source_xy", "map_xy")
        ray = find_ray_contributors(
            index,
            source_view_id,
            source_xy,
            radius_px=radius_px,
            quantization_px=quantization_px,
        )
        if ray is None or not ray.get("contributors"):
            dropped_no_contributors += 1
            continue

        query_id = str(match.get("query_id", "")).strip()
        if query_id:
            query_ids.add(query_id)
        source_view_ids.add(source_view_id)
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "split_name": split,
            "query_id": query_id,
            "source_view_id": source_view_id,
            "source_xy": source_xy,
            "ray_pixel_xy": list(ray["pixel_xy"]),
            "contributors": [dict(item) for item in ray["contributors"]],
            "pnp_inlier": bool(match.get("pnp_inlier", False)),
            "reprojection_error_px": _float(match.get("reprojection_error_px"), 0.0),
            "descriptor_score": _float(match.get("descriptor_score"), 1.0),
            "local_geometry_score": _float(match.get("local_geometry_score"), 1.0),
        }
        if "query_xy" in match:
            row["query_xy"] = _xy(match, "query_xy")
        for key, length in (
            ("query_xy_norm", 2),
            ("xy_norm", 2),
            ("normalized_xy", 2),
            ("bearing", 3),
            ("bearing_vector", 3),
            ("camera_xyz", 3),
            ("camera_point", 3),
            ("camera_frame_xyz", 3),
        ):
            vector = _vector(match, key, length=length)
            if vector is not None:
                row[key] = vector
        for key in ("descriptor_margin", "ray_entropy"):
            if key in match:
                row[key] = _float(match[key])
            elif key in ray:
                row[key] = _float(ray[key])
        for key in ("expected_depth", "rendered_depth", "artifact_score"):
            if key in match:
                row[key] = _float(match[key])
            elif key in ray:
                row[key] = _float(ray[key])
        observations.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "observations": observations,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "split_name": split,
            "match_count": int(len(matches)),
            "observation_count": int(len(observations)),
            "dropped_no_contributors": int(dropped_no_contributors),
            "query_count": int(len(query_ids)),
            "source_view_count": int(len(source_view_ids)),
            "radius_px": float(radius_px),
            "quantization_px": float(quantization_px),
        },
    }
