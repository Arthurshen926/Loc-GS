from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

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
        row["feature_map_path"] = str(_feature_map_path(row))
        if dry_run:
            row["render_status"] = "dry_run"
            row["rgb_exists"] = bool(Path(str(row.get("rgb_path", ""))).is_file())
            row["depth_exists"] = bool(Path(str(row.get("depth_path", ""))).is_file())
            row["feature_map_exists"] = bool(Path(str(row.get("feature_map_path", ""))).is_file())
            updated.append(row)
            continue
        try:
            rendered = render_fn(row)
            _write_rgb(row["rgb_path"], rendered["rgb"])
            if "depth" in rendered and row.get("depth_path"):
                _write_depth(row["depth_path"], rendered["depth"])
            wrote_feature_map = _maybe_write_feature_map(row, rendered)
            row["rgb_exists"] = bool(Path(str(row["rgb_path"])).is_file())
            row["depth_exists"] = bool(Path(str(row.get("depth_path", ""))).is_file())
            row["feature_map_exists"] = wrote_feature_map or bool(Path(str(row["feature_map_path"])).is_file())
            row["render_status"] = "rendered"
        except Exception as exc:
            failures += 1
            row["render_status"] = "failed"
            row["render_error"] = str(exc)
            row["rgb_exists"] = bool(Path(str(row.get("rgb_path", ""))).is_file())
            row["depth_exists"] = bool(Path(str(row.get("depth_path", ""))).is_file())
            row["feature_map_exists"] = bool(Path(str(row.get("feature_map_path", ""))).is_file())
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
        "rendered_feature_map_count": int(
            sum(1 for row in updated if bool(row.get("feature_map_exists")) and row.get("render_status") == "rendered")
        ),
        "existing_feature_map_count": int(sum(1 for row in updated if bool(row.get("feature_map_exists")))),
        "failed_render_count": int(failures),
        "render_engine": "3dgs",
        "render_contract": "posed_3dgs_camera_v1",
    }
    return summary, updated


def build_feature_map_cache_from_render_records(
    records: Sequence[Mapping[str, Any]],
    *,
    output_cache: str | Path,
    scene: str,
    split_name: str,
    max_records: int | None = None,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal 3DGS render feature map cache")
    selected = list(records)
    if max_records is not None:
        selected = selected[: max(0, int(max_records))]
    entries: list[dict[str, Any]] = []
    missing_feature_maps: list[str] = []
    descriptor_dim = 0
    for row_idx, record in enumerate(selected):
        row = dict(record)
        row_split = reject_test_split(
            str(row.get("split_name") or split),
            purpose="internal 3DGS render feature map cache row",
        )
        if row_split != split:
            raise ValueError(f"feature map cache split mismatch at row {row_idx}: expected {split}, got {row_split}")
        row_scene = str(row.get("scene") or scene)
        if row_scene != str(scene):
            raise ValueError(f"feature map cache scene mismatch at row {row_idx}: expected {scene}, got {row_scene}")
        path = _feature_map_path(row)
        synthetic_query_id = str(row.get("synthetic_query_id") or row.get("image_id") or "")
        if not path.is_file():
            missing_feature_maps.append(synthetic_query_id)
            continue
        feature_payload = _load_feature_map_payload(path)
        descriptor = torch.as_tensor(feature_payload["descriptor_map"], dtype=torch.float32).cpu()
        descriptor_dim = descriptor_dim or int(descriptor.shape[-3] if descriptor.ndim >= 3 else 0)
        entry = {
            "image_id": str(feature_payload.get("image_id") or synthetic_query_id),
            "descriptor_map": descriptor,
        }
        for key in ("score_map", "keypoints_yx", "keypoint_scores"):
            if key in feature_payload:
                entry[key] = torch.as_tensor(feature_payload[key]).cpu()
        entries.append(entry)
    output = {
        "metadata": {
            "schema_version": "internal_query_feature_map_cache_v1",
            "scene": str(scene),
            "split_name": split,
            "source": "internal_3dgs_render_feature_map_cache",
            "split_audit": {
                "schema_version": "internal_split_audit_v1",
                "audit_status": "passed",
                "split_name": split,
                "official_test_used": False,
                "test_split_used": False,
            },
        },
        "entries": entries,
    }
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return {
        "schema_version": "internal_render_feature_map_cache_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "output_cache": str(output_path),
        "record_count": int(len(selected)),
        "entry_count": int(len(entries)),
        "missing_feature_map_count": int(len(missing_feature_maps)),
        "missing_feature_map_ids_preview": missing_feature_maps[:10],
        "descriptor_dim": int(descriptor_dim),
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


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


def _feature_map_path(record: Mapping[str, Any]) -> Path:
    raw_path = record.get("feature_map_path")
    if raw_path:
        return Path(str(raw_path))
    rgb_path = record.get("rgb_path")
    if not rgb_path:
        raise ValueError("render record is missing feature_map_path and rgb_path")
    return Path(str(rgb_path)).with_suffix(".features.pt")


def _maybe_write_feature_map(record: Mapping[str, Any], rendered: Mapping[str, Any]) -> bool:
    descriptor = rendered.get("descriptor_map", rendered.get("feature_map"))
    if descriptor is None:
        return False
    target = _feature_map_path(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "internal_render_feature_map_record_v1",
        "image_id": str(record.get("synthetic_query_id") or record.get("image_id") or ""),
        "descriptor_map": torch.as_tensor(descriptor, dtype=torch.float32).detach().cpu(),
    }
    score_map = rendered.get("score_map", rendered.get("detector_score_map"))
    if score_map is not None:
        payload["score_map"] = torch.as_tensor(score_map, dtype=torch.float32).detach().cpu()
    for key in ("keypoints_yx", "query_yx", "keypoint_scores", "query_score"):
        if key not in rendered:
            continue
        output_key = "keypoint_scores" if key == "query_score" else "keypoints_yx" if key == "query_yx" else key
        payload[output_key] = torch.as_tensor(rendered[key]).detach().cpu()
    torch.save(payload, target)
    return True


def _load_feature_map_payload(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"feature map artifact must contain a mapping: {path}")
    descriptor = payload.get("descriptor_map", payload.get("feature_map"))
    if descriptor is None:
        raise ValueError(f"feature map artifact is missing descriptor_map: {path}")
    result = dict(payload)
    result["descriptor_map"] = descriptor
    return result


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
