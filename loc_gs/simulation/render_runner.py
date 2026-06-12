from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from loc_gs.sparse.audit import reject_test_split

RenderFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def render_assets_from_manifest_records(
    records: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    split_name: str,
    render_fn: RenderFn,
    dry_run: bool = False,
    max_records: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    split = reject_test_split(split_name, purpose="internal 3DGS render runner")
    selected = list(records)
    if max_records is not None:
        selected = selected[: max(0, int(max_records))]
    updated: list[dict[str, object]] = []
    failures = 0
    for row_idx, record in enumerate(selected):
        row = dict(record)
        row_split = reject_test_split(str(row.get("split_name") or split), purpose="internal 3DGS render runner row")
        if row_split != split:
            raise ValueError(f"render runner split mismatch at row {row_idx}: expected {split}, got {row_split}")
        row_scene = str(row.get("scene") or scene)
        if row_scene != str(scene):
            raise ValueError(f"render runner scene mismatch at row {row_idx}: expected {scene}, got {row_scene}")
        if dry_run:
            row["render_status"] = "dry_run"
            row["rgb_exists"] = bool(Path(str(row.get("rgb_path", ""))).is_file())
            row["depth_exists"] = bool(Path(str(row.get("depth_path", ""))).is_file())
            updated.append(row)
            continue
        try:
            rendered = render_fn(row)
            _write_rgb(row["rgb_path"], rendered["rgb"])
            if "depth" in rendered and row.get("depth_path"):
                _write_depth(row["depth_path"], rendered["depth"])
            row["rgb_exists"] = bool(Path(str(row["rgb_path"])).is_file())
            row["depth_exists"] = bool(Path(str(row.get("depth_path", ""))).is_file())
            row["render_status"] = "rendered"
        except Exception as exc:
            failures += 1
            row["render_status"] = "failed"
            row["render_error"] = str(exc)
            row["rgb_exists"] = bool(Path(str(row.get("rgb_path", ""))).is_file())
            row["depth_exists"] = bool(Path(str(row.get("depth_path", ""))).is_file())
        updated.append(row)
    summary = {
        "schema_version": "internal_render_runner_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "dry_run": bool(dry_run),
        "render_request_count": int(len(updated)),
        "rendered_rgb_count": int(
            sum(1 for row in updated if bool(row.get("rgb_exists")) and row.get("render_status") == "rendered")
        ),
        "existing_rgb_count": int(sum(1 for row in updated if bool(row.get("rgb_exists")))),
        "rendered_depth_count": int(
            sum(1 for row in updated if bool(row.get("depth_exists")) and row.get("render_status") == "rendered")
        ),
        "failed_render_count": int(failures),
        "render_engine": "3dgs",
        "render_contract": "posed_3dgs_camera_v1",
    }
    return summary, updated


def render_record_with_internal_3dgs(
    record: Mapping[str, Any],
    *,
    gaussian_ply: str | Path,
    device: str = "cuda",
    latent_dim: int = 16,
) -> dict[str, np.ndarray]:
    import torch

    from loc_gs.models.explicit_gaussian import ExplicitFeatureGaussian
    from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

    intr = record.get("render_intrinsics")
    if not isinstance(intr, Mapping):
        raise ValueError("render record is missing render_intrinsics")
    pose_c2w = np.asarray(record.get("render_pose_c2w"), dtype=np.float64).reshape(4, 4)
    pose_w2c = np.linalg.inv(pose_c2w)
    model = ExplicitFeatureGaussian(latent_dim=int(latent_dim))
    model.load_from_ply(str(gaussian_ply))
    target_device = torch.device(device)
    model.to(target_device)
    renderer = FeatureFieldRenderer(
        image_height=int(intr["height"]),
        image_width=int(intr["width"]),
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
    ).to(target_device)
    with torch.no_grad():
        result = renderer.render_rgb(
            model,
            torch.as_tensor(pose_w2c, dtype=torch.float32, device=target_device),
        )
    return {
        "rgb": result["rgb"].detach().cpu().permute(1, 2, 0).numpy(),
        "depth": result["depth"].detach().cpu().numpy(),
    }


def load_render_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"render manifest row {line_number} must be a JSON object")
        rows.append(row)
    return rows


def _write_rgb(path: str | Path, rgb: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise ValueError(f"rgb render must have shape HxWx3 or HxWx4, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is expected in the project environment.
        raise RuntimeError("Pillow is required to write RGB render assets") from exc
    Image.fromarray(arr).save(target)


def _write_depth(path: str | Path, depth: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(depth, dtype=np.float32))
