from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from loc_gs.sparse.audit import reject_test_split


def build_render_manifest_from_plan_rows(
    plan_rows: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    split_name: str,
    render_root: str | Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = reject_test_split(split_name, purpose="internal 3DGS render manifest")
    root = Path(render_root)
    records: list[dict[str, object]] = []
    for row_idx, row in enumerate(plan_rows):
        row_split = reject_test_split(str(row.get("split_name") or split), purpose="internal 3DGS render manifest row")
        if row_split != split:
            raise ValueError(f"render manifest split mismatch at row {row_idx}: expected {split}, got {row_split}")
        row_scene = str(row.get("scene") or scene)
        if row_scene != str(scene):
            raise ValueError(f"render manifest scene mismatch at row {row_idx}: expected {scene}, got {row_scene}")
        synthetic_query_id = str(row.get("synthetic_query_id") or "")
        if not synthetic_query_id:
            raise ValueError(f"simulation plan row {row_idx} is missing synthetic_query_id")
        rgb_path = root / f"{synthetic_query_id}.png"
        depth_path = root / f"{synthetic_query_id}.depth.npy"
        records.append(
            {
                "schema_version": "internal_render_manifest_record_v1",
                "scene": str(scene),
                "split_name": split,
                "synthetic_query_id": synthetic_query_id,
                "source_image_id": str(row.get("source_image_id") or ""),
                "render_contract": str(row.get("render_contract") or "posed_3dgs_camera_v1"),
                "render_engine": str(row.get("render_engine") or "3dgs"),
                "render_pose_c2w": row.get("render_pose_c2w"),
                "render_intrinsics": row.get("render_intrinsics"),
                "rgb_path": str(rgb_path),
                "rgb_exists": bool(rgb_path.is_file()),
                "depth_path": str(depth_path),
                "depth_exists": bool(depth_path.is_file()),
                "role": "training_render_asset",
                "dense_inference_enabled": False,
                "external_runtime_dependency": "forbidden",
            }
        )
    missing_rgb = [record for record in records if not bool(record["rgb_exists"])]
    missing_depth = [record for record in records if not bool(record["depth_exists"])]
    summary = {
        "schema_version": "internal_render_manifest_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "render_root": str(root),
        "render_request_count": int(len(records)),
        "rendered_rgb_count": int(len(records) - len(missing_rgb)),
        "missing_rgb_count": int(len(missing_rgb)),
        "rendered_depth_count": int(len(records) - len(missing_depth)),
        "missing_depth_count": int(len(missing_depth)),
        "render_engine": "3dgs",
        "render_contract": "posed_3dgs_camera_v1",
    }
    return summary, records


def load_simulation_plan_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"simulation plan row {line_number} must be a JSON object")
        rows.append(row)
    return rows
