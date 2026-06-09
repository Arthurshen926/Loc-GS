from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name


SCHEMA_VERSION = "projected_gaussian_rays_v1"


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
        raise ValueError("coordinate field must be a sequence")
    values = list(value)
    if len(values) != 2:
        raise ValueError(f"coordinate field must contain 2 values, got {len(values)}")
    return [_float(values[0]), _float(values[1])]


def _gaussian_id(record: Mapping[str, Any]) -> int:
    value = record.get("gaussian_id", record.get("matched_gaussian_id"))
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid gaussian_id: {value!r}") from exc
    if out < 0:
        raise ValueError(f"gaussian_id must be non-negative, got {out}")
    return out


def proxy_contribution(*, distance_px: float, radius_px: float, opacity: float = 1.0) -> float:
    radius = max(_float(radius_px, 1.0), 1.0)
    sigma = max(radius, 1.0)
    distance = max(0.0, _float(distance_px))
    alpha = max(0.0, min(1.0, _float(opacity, 1.0)))
    return float(alpha * math.exp(-0.5 * (distance / sigma) ** 2))


def _projection_source(record: Mapping[str, Any]) -> str:
    return str(record.get("source_view_id", record.get("view_id", ""))).strip()


def _record_split(record: Mapping[str, Any], default: str) -> str:
    split = str(record.get("split_name", record.get("split", default))).strip() or default
    return validate_split_name(split)


def projected_gaussians_to_sparse_source_rays(
    matches: Sequence[Mapping[str, Any]],
    projections: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    top_k: int = 8,
    max_radius_px: float = 8.0,
    radius_scale: float = 2.0,
    min_contribution: float = 1e-6,
    require_full_raw_source: bool = True,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    cell_size = max(1.0, float(max_radius_px))
    projection_grid: dict[str, dict[tuple[int, int], list[dict[str, Any]]]] = {}
    for raw in projections:
        _record_split(raw, split)
        projection_source = str(raw.get("projection_source", "")).strip()
        if bool(require_full_raw_source) and projection_source != "full_raw_gaussians":
            raise ValueError("projected Gaussian rays require projection_source=full_raw_gaussians")
        source = _projection_source(raw)
        if not source:
            continue
        xy = _xy(raw, "xy", "means2d", "pixel_xy", "source_xy")
        row = dict(raw)
        row["source_view_id"] = source
        row["xy"] = xy
        row["gaussian_id"] = _gaussian_id(raw)
        row["projection_source"] = projection_source or "unknown"
        row["radius"] = max(_float(raw.get("radius", raw.get("radii", 1.0)), 1.0), 1.0)
        row["opacity"] = max(0.0, min(1.0, _float(raw.get("opacity", raw.get("alpha", 1.0)), 1.0)))
        row["depth"] = _float(raw.get("depth", raw.get("z", 0.0)))
        cell = (int(math.floor(xy[0] / cell_size)), int(math.floor(xy[1] / cell_size)))
        projection_grid.setdefault(source, {}).setdefault(cell, []).append(row)

    rays: list[dict[str, Any]] = []
    dropped_no_projection = 0
    seen_pixels: set[tuple[str, int, int]] = set()
    for match in matches:
        _record_split(match, split)
        source = str(match.get("source_view_id", match.get("view_id", ""))).strip()
        if not source:
            raise ValueError("match record is missing source_view_id")
        source_xy = _xy(match, "source_xy", "query_xy", "xy")
        key = (source, int(round(source_xy[0])), int(round(source_xy[1])))
        if key in seen_pixels:
            continue
        seen_pixels.add(key)

        candidates: list[dict[str, Any]] = []
        source_grid = projection_grid.get(source, {})
        center_cell = (int(math.floor(source_xy[0] / cell_size)), int(math.floor(source_xy[1] / cell_size)))
        nearby_projections: list[dict[str, Any]] = []
        for cell_x in range(center_cell[0] - 1, center_cell[0] + 2):
            for cell_y in range(center_cell[1] - 1, center_cell[1] + 2):
                nearby_projections.extend(source_grid.get((cell_x, cell_y), []))
        for projection in nearby_projections:
            pxy = projection["xy"]
            distance = math.hypot(float(pxy[0]) - source_xy[0], float(pxy[1]) - source_xy[1])
            radius = float(projection["radius"])
            gate = min(float(max_radius_px), max(1.0, float(radius_scale) * radius))
            if distance > gate:
                continue
            contribution = proxy_contribution(
                distance_px=distance,
                radius_px=radius,
                opacity=float(projection["opacity"]),
            )
            if contribution < float(min_contribution):
                continue
            item: dict[str, Any] = {
                "gaussian_id": int(projection["gaussian_id"]),
                "contribution": float(contribution),
                "depth": float(projection["depth"]),
                "reliability": max(0.0, min(1.0, 1.0 - distance / max(gate, 1e-6))),
            }
            if "xyz" in projection:
                xyz = list(projection["xyz"])
                if len(xyz) == 3:
                    item["xyz"] = [_float(xyz[0]), _float(xyz[1]), _float(xyz[2])]
            candidates.append(item)

        if not candidates:
            dropped_no_projection += 1
            continue

        selected = sorted(candidates, key=lambda item: (-float(item["contribution"]), int(item["gaussian_id"])))[: int(top_k)]
        weights = [float(item["contribution"]) for item in selected]
        depths = [float(item.get("depth", 0.0)) for item in selected]
        weight_sum = sum(weights)
        expected_depth = (
            sum(depth * weight for depth, weight in zip(depths, weights)) / max(weight_sum, 1e-12)
            if weight_sum > 0.0 and any(depth > 0.0 for depth in depths)
            else 0.0
        )
        positive_depths = [depth for depth in depths if depth > 0.0]
        rays.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_view_id": source,
                "split_name": split,
                "pixel_xy": [float(source_xy[0]), float(source_xy[1])],
                "contributors": selected,
                "rendered_depth": float(min(positive_depths)) if positive_depths else 0.0,
                "expected_depth": float(expected_depth),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "rays": rays,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "split_name": split,
            "match_count": int(len(matches)),
            "projection_count": int(len(projections)),
            "ray_count": int(len(rays)),
            "dropped_no_projection": int(dropped_no_projection),
            "top_k": int(top_k),
            "max_radius_px": float(max_radius_px),
            "radius_scale": float(radius_scale),
            "weight_source": "projected_gaussian_radius_opacity_proxy",
            "projection_source": "full_raw_gaussians" if bool(require_full_raw_source) else "mixed_or_unknown",
            "require_full_raw_source": bool(require_full_raw_source),
            "spatial_index_cell_size_px": float(cell_size),
            "spatial_index_source_count": int(len(projection_grid)),
        },
    }
