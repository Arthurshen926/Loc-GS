from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from loc_gs.feedback.ray_attributed_solver_feedback import validate_split_name


SCHEMA_VERSION = "raster_ray_contributors_v1"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _gaussian_id(record: Mapping[str, Any]) -> int:
    value = record.get("gaussian_id", record.get("matched_gaussian_id"))
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid gaussian_id: {value!r}") from exc
    if out < 0:
        raise ValueError(f"gaussian_id must be non-negative, got {out}")
    return out


def infer_pixel_xy(record: Mapping[str, Any], *, width: int) -> list[float]:
    if "pixel_xy" in record:
        value = record["pixel_xy"]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("pixel_xy must be a sequence")
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"pixel_xy must contain 2 values, got {len(values)}")
        return [_float(values[0]), _float(values[1])]
    if "xy" in record:
        return infer_pixel_xy({"pixel_xy": record["xy"]}, width=width)
    if "pixel_id" not in record:
        raise ValueError("intersection record must contain pixel_xy or pixel_id")
    image_width = int(width)
    if image_width <= 0:
        raise ValueError(f"width must be positive when pixel_id is used, got {width}")
    pixel_id = int(record["pixel_id"])
    if pixel_id < 0:
        raise ValueError(f"pixel_id must be non-negative, got {pixel_id}")
    return [float(pixel_id % image_width), float(pixel_id // image_width)]


def _contribution(record: Mapping[str, Any]) -> float:
    return max(
        0.0,
        _float(
            record.get(
                "contribution",
                record.get("weight", record.get("alpha", record.get("opacity", 0.0))),
            )
        ),
    )


def _pixel_key(xy: Sequence[float]) -> tuple[int, int]:
    return (int(round(float(xy[0]))), int(round(float(xy[1]))))


def raster_intersections_to_ray_records(
    intersections: Sequence[Mapping[str, Any]],
    *,
    source_view_id: str,
    split_name: str,
    width: int,
    top_k: int = 8,
    min_contribution: float = 0.0,
) -> dict[str, Any]:
    split = validate_split_name(split_name)
    source = str(source_view_id).strip()
    if not source:
        raise ValueError("source_view_id is required")
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in intersections:
        record_split = str(raw.get("split_name", raw.get("split", split))).strip() or split
        validate_split_name(record_split)
        pixel_xy = infer_pixel_xy(raw, width=int(width))
        contribution = _contribution(raw)
        if contribution < float(min_contribution):
            continue
        item: dict[str, Any] = {
            "gaussian_id": _gaussian_id(raw),
            "contribution": float(contribution),
        }
        for key in ("depth", "z", "reliability", "visibility", "opacity"):
            if key in raw:
                item[key] = _float(raw[key])
        if "xyz" in raw:
            xyz = list(raw["xyz"])
            if len(xyz) != 3:
                raise ValueError("xyz must contain 3 values")
            item["xyz"] = [_float(xyz[0]), _float(xyz[1]), _float(xyz[2])]
        grouped[_pixel_key(pixel_xy)].append({"pixel_xy": pixel_xy, **item})

    rays: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda item: (-float(item["contribution"]), int(item["gaussian_id"])))
        selected = items[: int(top_k)]
        contributors = [{k: v for k, v in item.items() if k != "pixel_xy"} for item in selected]
        depths = [float(item.get("depth", item.get("z", 0.0))) for item in selected]
        weights = [float(item["contribution"]) for item in selected]
        positive_depths = [depth for depth in depths if depth > 0.0]
        weight_sum = sum(weights)
        expected_depth = (
            sum(depth * weight for depth, weight in zip(depths, weights)) / max(weight_sum, 1e-12)
            if weight_sum > 0.0 and any(depth > 0.0 for depth in depths)
            else 0.0
        )
        rays.append(
            {
                "schema_version": SCHEMA_VERSION,
                "source_view_id": source,
                "split_name": split,
                "pixel_xy": [float(key[0]), float(key[1])],
                "contributors": contributors,
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
            "source_view_id": source,
            "intersection_count": int(len(intersections)),
            "ray_count": int(len(rays)),
            "top_k": int(top_k),
            "min_contribution": float(min_contribution),
            "weight_source": "contribution_or_opacity_proxy",
        },
    }
