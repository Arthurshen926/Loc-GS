from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from loc_gs.feedback.audit import audit_feedback_bank_v3
from loc_gs.feedback.io import save_feedback_bank, summarize_feedback_bank
from loc_gs.feedback.schema import FeedbackMatchRecord


SCHEMA_VERSION = "feedback_bank_v3"
POSE_SOURCE = "ulfloc_sparse_pnp_train_selfmap"


def validate_split_name(split_name: str) -> str:
    normalized = str(split_name).strip()
    if not normalized:
        raise ValueError("split_name is required")
    if normalized.lower() == "test":
        raise ValueError("refusing to export ULF-Loc sparse feedback from test split")
    return normalized


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    array = np.asarray(value)
    if array.ndim == 0:
        return [array.item()]
    return array.reshape(-1).tolist()


def _xy_list(value: Any) -> list[tuple[float | None, float | None]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        return []
    array = array.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in array]


def _xyz_list(value: Any) -> list[tuple[float | None, float | None, float | None]]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        return []
    array = array.reshape(-1, 3)
    return [(float(x), float(y), float(z)) for x, y, z in array]


def _matrix_array(value: Any, *, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    try:
        array = array.astype(np.float64).reshape(shape)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(array).all():
        return None
    return array


def _image_size(details: dict[str, Any]) -> tuple[float, float] | None:
    raw = details.get("image_size")
    if raw is None:
        return None
    values = np.asarray(raw).reshape(-1).tolist()
    if len(values) < 2:
        return None
    try:
        width = float(values[0])
        height = float(values[1])
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def _sparse_geometry_observation(
    *,
    query_xy: tuple[float | None, float | None],
    p3d: tuple[float | None, float | None, float | None] | None,
    intrinsic: np.ndarray | None,
    pose_w2c: np.ndarray | None,
    image_size: tuple[float, float] | None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    if image_size is not None and query_xy[0] is not None and query_xy[1] is not None:
        width, height = image_size
        observation["query_xy_norm"] = [float(query_xy[0]) / width, float(query_xy[1]) / height]
    if intrinsic is not None and query_xy[0] is not None and query_xy[1] is not None:
        pixel = np.asarray([float(query_xy[0]), float(query_xy[1]), 1.0], dtype=np.float64)
        try:
            bearing = np.linalg.solve(intrinsic, pixel)
        except np.linalg.LinAlgError:
            bearing = None
        if bearing is not None and np.isfinite(bearing).all():
            observation["bearing"] = [float(value) for value in bearing.reshape(-1)[:3]]
    if pose_w2c is not None and p3d is not None:
        if p3d[0] is not None and p3d[1] is not None and p3d[2] is not None:
            point_h = np.asarray([float(p3d[0]), float(p3d[1]), float(p3d[2]), 1.0], dtype=np.float64)
            camera_xyz = pose_w2c @ point_h
            if np.isfinite(camera_xyz).all():
                camera_values = [float(value) for value in camera_xyz.reshape(-1)[:3]]
                observation["camera_xyz"] = camera_values
                observation["depth_m"] = float(camera_values[2])
    return observation


def _image_cell_from_query_xy_norm(query_xy_norm: Any, *, grid_size: int = 8) -> tuple[int, int] | None:
    if not isinstance(query_xy_norm, (list, tuple)) or len(query_xy_norm) < 2:
        return None
    try:
        x = float(query_xy_norm[0])
        y = float(query_xy_norm[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x, y]).all():
        return None
    grid = max(1, int(grid_size))
    return min(grid - 1, max(0, int(x * grid))), min(grid - 1, max(0, int(y * grid)))


def _depth_bin_from_depth_m(depth_m: Any) -> int | None:
    depth = _to_float(depth_m)
    if depth is None or not np.isfinite(depth) or depth <= 0.0:
        return -1
    if depth < 2.0:
        return 0
    if depth < 5.0:
        return 1
    if depth < 10.0:
        return 2
    return 3


def _get_sequence(sequences: dict[str, list[Any]], key: str, index: int, default: Any = None) -> Any:
    values = sequences.get(key, [])
    return values[index] if index < len(values) else default


_SPARSE_DETAIL_SEQUENCE_KEYS = (
    "matched_sampled_indices",
    "matched_gaussian_ids",
    "query_keypoint_indices",
    "keypoint_xy",
    "query_xy",
    "p3d",
    "descriptor_margin",
    "detector_score",
    "descriptor_score",
    "pnp_inlier_mask",
    "reprojection_error_px",
)


def _sequence_length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().shape[0]) if value.ndim > 0 else 1
    array = np.asarray(value)
    if array.ndim == 0:
        return 1
    return int(array.shape[0])


def _filter_sequence(value: Any, keep_indices: list[int]) -> Any:
    if value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        index = torch.as_tensor(keep_indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    filtered = array[keep_indices]
    return filtered.tolist()


def compact_ulfloc_sparse_details(
    details: dict[str, Any],
    *,
    inliers_only: bool = False,
    max_non_inlier_records: int = 0,
) -> dict[str, Any]:
    """Return a compact sparse trace suitable for solver-aware sampling.

    Full ULF matching emits one record for every query keypoint match. For
    sparse solver feedback the useful positive support is the PnP inlier set;
    optional high-score non-inliers can be kept as ambiguity/hard-negative
    evidence without writing millions of records.
    """

    if not bool(inliers_only) and int(max_non_inlier_records) <= 0:
        return dict(details)

    count = max((_sequence_length(details.get(key)) for key in _SPARSE_DETAIL_SEQUENCE_KEYS), default=0)
    if count <= 0:
        return dict(details)
    inlier_mask = _as_list(details.get("pnp_inlier_mask"))
    inlier_mask = [bool(value) for value in inlier_mask[:count]]
    if len(inlier_mask) < count:
        inlier_mask.extend([False] * (count - len(inlier_mask)))

    keep = {idx for idx, is_inlier in enumerate(inlier_mask) if is_inlier}
    non_inlier_budget = max(0, int(max_non_inlier_records))
    if non_inlier_budget > 0:
        scores = _as_list(details.get("descriptor_score"))
        scored: list[tuple[float, int]] = []
        for idx in range(count):
            if idx in keep:
                continue
            score = 0.0
            if idx < len(scores):
                try:
                    score = float(scores[idx])
                except (TypeError, ValueError):
                    score = 0.0
            scored.append((-score, idx))
        for _, idx in sorted(scored)[:non_inlier_budget]:
            keep.add(int(idx))

    keep_indices = sorted(keep)
    compact = dict(details)
    for key in _SPARSE_DETAIL_SEQUENCE_KEYS:
        if key in compact and _sequence_length(compact.get(key)) == count:
            compact[key] = _filter_sequence(compact.get(key), keep_indices)
    compact["sparse_total_match_count"] = int(count)
    compact["sparse_retained_match_count"] = int(len(keep_indices))
    compact["sparse_compaction"] = {
        "inliers_only": bool(inliers_only),
        "max_non_inlier_records": int(non_inlier_budget),
    }
    return compact


def records_from_ulfloc_sparse_details(
    *,
    scene: str,
    image_id: str,
    details: dict[str, Any],
    split_name: str,
    source_role: str = "baseline_trace",
) -> list[FeedbackMatchRecord]:
    """Convert one ULF-Loc sparse localization trace into feedback-bank records."""
    split = validate_split_name(split_name)
    sequences = {
        "matched_sampled_indices": _as_list(details.get("matched_sampled_indices")),
        "matched_gaussian_ids": _as_list(details.get("matched_gaussian_ids")),
        "query_keypoint_indices": _as_list(details.get("query_keypoint_indices")),
        "detector_score": _as_list(details.get("detector_score")),
        "descriptor_score": _as_list(details.get("descriptor_score")),
        "pnp_inlier_mask": _as_list(details.get("pnp_inlier_mask")),
        "reprojection_error_px": _as_list(details.get("reprojection_error_px")),
    }
    keypoint_xy = _xy_list(details.get("keypoint_xy"))
    query_xy_values = _xy_list(details.get("query_xy"))
    p3d_values = _xyz_list(details.get("p3d"))
    intrinsic = _matrix_array(details.get("intrinsic"), shape=(3, 3))
    pose_w2c = _matrix_array(details.get("pose_w2c", details.get("selfmap_gt_w2c")), shape=(4, 4))
    image_size = _image_size(details)
    descriptor_margins = _as_list(details.get("descriptor_margin"))
    count = max(
        len(sequences["matched_sampled_indices"]),
        len(sequences["matched_gaussian_ids"]),
        len(keypoint_xy),
    )
    records: list[FeedbackMatchRecord] = []
    for idx in range(count):
        sampled_index = _get_sequence(sequences, "matched_sampled_indices", idx, idx)
        gaussian_id = _get_sequence(sequences, "matched_gaussian_ids", idx, sampled_index)
        keypoint_index = _get_sequence(sequences, "query_keypoint_indices", idx, idx)
        xy = keypoint_xy[idx] if idx < len(keypoint_xy) else (None, None)
        query_xy = query_xy_values[idx] if idx < len(query_xy_values) else xy
        p3d = p3d_values[idx] if idx < len(p3d_values) else None
        geometry = _sparse_geometry_observation(
            query_xy=query_xy,
            p3d=p3d,
            intrinsic=intrinsic,
            pose_w2c=pose_w2c,
            image_size=image_size,
        )
        query_xy_norm = tuple(geometry["query_xy_norm"]) if "query_xy_norm" in geometry else None
        depth_m = _to_float(geometry.get("depth_m"))
        records.append(
            FeedbackMatchRecord(
                scene=str(scene),
                split_name=split,
                query_id=str(image_id),
                image_id=str(image_id),
                source_view_id=str(image_id),
                pose_source=POSE_SOURCE,
                source_role=str(source_role),
                keypoint_id=f"kp:{keypoint_index}",
                keypoint_xy=xy,
                matched_landmark_id=f"sampled:{sampled_index}",
                matched_gaussian_id=str(gaussian_id),
                descriptor_score=_to_float(_get_sequence(sequences, "descriptor_score", idx)),
                detector_score=_to_float(_get_sequence(sequences, "detector_score", idx)),
                match_rank=0,
                pnp_inlier=bool(_get_sequence(sequences, "pnp_inlier_mask", idx, False)),
                reprojection_error_px=_to_float(_get_sequence(sequences, "reprojection_error_px", idx)),
                pose_error_t_cm=_to_float(details.get("pose_error_t_cm")),
                query_sparse_te_cm=_to_float(details.get("pose_error_t_cm")),
                pose_error_r_deg=_to_float(details.get("pose_error_r_deg")),
                pnp_success=bool(details.get("pnp_success", False)),
                pose_success=bool(details.get("pnp_success", False)),
                dense_refine_success=False,
                dense_transition="not_run_sparse_feedback",
                dense_delta_te_cm=0.0,
                query_xy_norm=query_xy_norm,
                bearing=tuple(geometry["bearing"]) if "bearing" in geometry else None,
                camera_xyz=tuple(geometry["camera_xyz"]) if "camera_xyz" in geometry else None,
                depth_m=depth_m,
                descriptor_margin=_to_float_default(
                    _get_sequence({"descriptor_margin": descriptor_margins}, "descriptor_margin", idx),
                    0.0,
                ),
                local_geometry_score=1.0,
                image_cell=_image_cell_from_query_xy_norm(query_xy_norm),
                depth_bin=_depth_bin_from_depth_m(depth_m),
            )
        )
    return records


def cross_view_matches_from_ulfloc_sparse_details(
    *,
    scene: str,
    image_id: str,
    details: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    """Convert ULF sparse details into Phase 2A cross-view match rows.

    For train/self-map feedback, the source view is the same train image that is
    localized. The sparse keypoint coordinate is therefore also the source ray
    pixel used to join against rendered contributor records.
    """

    split = validate_split_name(split_name)
    sequences = {
        "matched_sampled_indices": _as_list(details.get("matched_sampled_indices")),
        "matched_gaussian_ids": _as_list(details.get("matched_gaussian_ids")),
        "query_keypoint_indices": _as_list(details.get("query_keypoint_indices")),
        "detector_score": _as_list(details.get("detector_score")),
        "descriptor_score": _as_list(details.get("descriptor_score")),
        "descriptor_margin": _as_list(details.get("descriptor_margin")),
        "pnp_inlier_mask": _as_list(details.get("pnp_inlier_mask")),
        "reprojection_error_px": _as_list(details.get("reprojection_error_px")),
    }
    keypoint_xy = _xy_list(details.get("keypoint_xy"))
    query_xy_values = _xy_list(details.get("query_xy"))
    p3d_values = _xyz_list(details.get("p3d"))
    intrinsic = _matrix_array(details.get("intrinsic"), shape=(3, 3))
    pose_w2c = _matrix_array(details.get("pose_w2c", details.get("selfmap_gt_w2c")), shape=(4, 4))
    image_size = _image_size(details)
    count = max(
        len(sequences["matched_sampled_indices"]),
        len(sequences["matched_gaussian_ids"]),
        len(keypoint_xy),
    )
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        sampled_index = _get_sequence(sequences, "matched_sampled_indices", idx, idx)
        gaussian_id = _get_sequence(sequences, "matched_gaussian_ids", idx, sampled_index)
        keypoint_index = _get_sequence(sequences, "query_keypoint_indices", idx, idx)
        xy = keypoint_xy[idx] if idx < len(keypoint_xy) else (None, None)
        query_xy = query_xy_values[idx] if idx < len(query_xy_values) else xy
        if xy[0] is None or xy[1] is None:
            continue
        row = {
            "schema_version": "ulfloc_sparse_cross_view_matches_v1",
            "scene": str(scene),
            "split_name": split,
            "teacher_source": "ulfloc_sampled_landmarks",
            "feedback_role": "bootstrap_sparse_teacher",
            "paper_safe_role": "diagnostic_not_main_full_raw_feedback",
            "query_id": str(image_id),
            "source_view_id": str(image_id),
            "query_keypoint_index": int(keypoint_index),
            "query_xy": [float(query_xy[0]), float(query_xy[1])],
            "source_xy": [float(xy[0]), float(xy[1])],
            "matched_sampled_index": int(sampled_index),
            "matched_gaussian_id": int(gaussian_id),
            "descriptor_score": _to_float(_get_sequence(sequences, "descriptor_score", idx)) or 0.0,
            "detector_score": _to_float(_get_sequence(sequences, "detector_score", idx)) or 0.0,
            "local_geometry_score": 1.0,
            "pnp_inlier": bool(_get_sequence(sequences, "pnp_inlier_mask", idx, False)),
            "reprojection_error_px": _to_float(_get_sequence(sequences, "reprojection_error_px", idx)) or 0.0,
        }
        descriptor_margin = _to_float(_get_sequence(sequences, "descriptor_margin", idx))
        if descriptor_margin is not None:
            row["descriptor_margin"] = float(descriptor_margin)
        p3d = p3d_values[idx] if idx < len(p3d_values) else None
        row.update(
            _sparse_geometry_observation(
                query_xy=query_xy,
                p3d=p3d,
                intrinsic=intrinsic,
                pose_w2c=pose_w2c,
                image_size=image_size,
            )
        )
        rows.append(row)
    return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _to_float_default(value: Any, default: float) -> float:
    parsed = _to_float(value)
    return float(default) if parsed is None else float(parsed)


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        return "unknown"


def _git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=str(cwd), text=True)
    except Exception as exc:
        return f"git status unavailable: {exc}\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_cross_view_matches_jsonl(path: Path, matches: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    rows = [{"type": "manifest", "manifest": dict(metadata)}]
    rows.extend({"type": "match", "match": dict(record)} for record in matches)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _load_masks(dataset_source: Path, images: str) -> dict[str, Any] | None:
    mask_path = dataset_source / images / "masks.pkl"
    if not mask_path.exists():
        return None
    with mask_path.open("rb") as handle:
        return pickle.load(handle)


def resolve_ulfloc_source_path_for_loader(source_path: Path, link_root: Path) -> Path:
    """Return a ULF loader-compatible source path without changing data identity.

    The upstream ULF loader checks dataset type with case-sensitive path
    substrings. Local Cambridge data roots can be named ``Cambridge_stdloc``;
    this helper creates a lowercase symlink for the loader while manifests keep
    the original path for audit.
    """

    source = Path(source_path)
    text = str(source)
    if "7scenes" in text or "12scenes" in text or "cambridge" in text:
        return source
    if "Cambridge" not in text and "CAMBRIDGE" not in text:
        return source

    link_root = Path(link_root)
    link_root.mkdir(parents=True, exist_ok=True)
    link_name = "cambridge_" + source.name
    link_path = link_root / link_name
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() != source.resolve():
            raise FileExistsError(f"ULF loader compatibility link points elsewhere: {link_path}")
    else:
        link_path.symlink_to(source.resolve(), target_is_directory=True)
    return link_path


def load_dataset_image_list(path: Path) -> list[str]:
    image_names: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        if first.lower().endswith((".png", ".jpg", ".jpeg")):
            image_names.append(first)
    return image_names


def resolve_train_images_to_read(source_path: Path, train_list: Path | None = None) -> tuple[list[str] | None, str | None]:
    candidates: list[Path] = []
    if train_list is not None:
        candidates.append(Path(train_list))
    candidates.extend(
        [
            Path(source_path) / "dataset_train.txt",
            Path(source_path) / "sfm_gt" / "list_train.txt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            image_names = load_dataset_image_list(candidate)
            if image_names:
                return image_names, str(candidate)
    return None, None


def _resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    resized = F.interpolate(mask.float(), size=(height, width), mode="nearest")
    return resized[0].bool()


def _apply_ulfloc_masks(query_image: torch.Tensor, masks: dict[str, Any] | None, image_name: str) -> torch.Tensor:
    if masks is None or image_name not in masks:
        return query_image
    height, width = query_image.shape[-2:]
    scene_masks = masks[image_name]
    obj_mask = _resize_mask(torch.as_tensor(scene_masks[0], device=query_image.device), height, width)[0]
    sky_mask = _resize_mask(torch.as_tensor(scene_masks[1], device=query_image.device), height, width)[0]
    distort_mask = _resize_mask(torch.as_tensor(scene_masks[2], device=query_image.device), height, width)[0]
    stable_mask = obj_mask & distort_mask
    masked = query_image * stable_mask[None]
    masked = masked.clone()
    masked[sky_mask[None].repeat(3, 1, 1) == False] = 0
    return masked


def _import_ulfloc(ulf_root: Path) -> dict[str, Any]:
    if str(ulf_root) not in sys.path:
        sys.path.insert(0, str(ulf_root))
    from scene import Scene  # type: ignore
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs  # type: ignore
    from ulfloc import ULFLoc  # type: ignore
    from utils.pose_utils import cal_pose_error  # type: ignore

    return {
        "Scene": Scene,
        "GaussianModel": GaussianModel,
        "GaussianModel_2dgs": GaussianModel_2dgs,
        "ULFLoc": ULFLoc,
        "cal_pose_error": cal_pose_error,
    }


def _dataset_namespace(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        sh_degree=int(args.sh_degree),
        source_path=os.path.abspath(str(args.source_path)),
        feature_type=str(args.feature_type or config.get("feature_type", "sp")),
        gaussian_type=str(args.gaussian_type),
        model_path=os.path.abspath(str(args.model_path)),
        images=str(args.images),
        resolution=int(args.resolution),
        white_background=True,
        longest_edge=int(args.longest_edge),
        data_device=str(args.data_device),
        eval=False,
        speedup=False,
        norm_before_render=True,
        render_items=["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    )


def configure_sparse_feedback_eval_overrides(
    config: dict[str, Any],
    *,
    model_path: Path,
    input_log_dir: Path | None,
    scene_detector_checkpoint: Path | None,
    scene_detector_blend_alpha: float,
    scene_detector_candidate_top_k: int,
    scene_detector_native_keep_fraction: float,
) -> Path:
    """Apply sparse-eval-only overrides when exporting feedback traces.

    Feedback attribution must match the sparse pipeline being validated.  In
    particular, solver-fused descriptor experiments use an external ULF log
    alias, and scene-detector experiments rerank SuperPoint before PnP.  Without
    these overrides the feedback bank describes a different candidate.
    """

    output_path = Path(input_log_dir) if input_log_dir is not None else Path(model_path) / str(config["log_name"])
    if scene_detector_checkpoint is not None:
        sparse_k = int(config.get("sparse", {}).get("kpts_num", 2048))
        candidate_top_k = (
            int(scene_detector_candidate_top_k)
            if int(scene_detector_candidate_top_k) > 0
            else sparse_k * 2
        )
        config["scene_specific_detector"] = {
            "enabled": True,
            "checkpoint": str(Path(scene_detector_checkpoint).resolve()),
            "mode": "rerank_superpoint",
            "candidate_top_k": int(candidate_top_k),
            "blend_alpha": float(scene_detector_blend_alpha),
            "native_keep_fraction": float(scene_detector_native_keep_fraction),
            "descriptor_stride": 8,
            "nms_radius": int(config.get("sparse", {}).get("nms", 4)),
            "score_threshold": 0.0,
        }
    else:
        config["scene_specific_detector"] = {"enabled": False}
    return output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ULF-Loc native sparse self-map feedback bank.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--input_log_dir", default=None, type=Path)
    parser.add_argument("--ulf_root", default="/root/ULF-Loc", type=Path)
    parser.add_argument("--split_name", default="selfmap_train")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--resolution", default=-1, type=int)
    parser.add_argument("--longest_edge", default=640, type=int)
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--compute_device", default="cuda")
    parser.add_argument("--scene_detector_checkpoint", default=None, type=Path)
    parser.add_argument("--scene_detector_blend_alpha", default=0.5, type=float)
    parser.add_argument("--scene_detector_candidate_top_k", default=0, type=int)
    parser.add_argument("--scene_detector_native_keep_fraction", default=0.0, type=float)
    parser.add_argument("--max_images", default=0, type=int)
    parser.add_argument("--inliers_only_for_bank", action="store_true")
    parser.add_argument("--max_non_inlier_records_per_image", default=0, type=int)
    parser.add_argument(
        "--source_role",
        default="baseline_trace",
        choices=("baseline_trace", "candidate_trace", "render_aug_trace"),
        help="Role stored in feedback_bank_v3 correspondence records.",
    )
    parser.add_argument("--no_masks", action="store_true")
    parser.add_argument("--train_list", default=None, type=Path)
    parser.add_argument(
        "--camera_split",
        choices=("train", "test"),
        default="train",
        help=(
            "Camera container to iterate after loading images_to_read. "
            "Use test only for train-dev validation overlays, never for official Cambridge test feedback."
        ),
    )
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()
    split_name = validate_split_name(args.split_name)
    camera_split = str(args.camera_split)
    if camera_split == "test":
        lowered_split = split_name.lower()
        if args.train_list is None:
            raise ValueError("--camera_split test requires --train_list to avoid accidental official-test feedback")
        if not any(token in lowered_split for token in ("train", "dev", "validation", "selfmap")):
            raise ValueError("--camera_split test is only allowed for explicitly named train/dev/validation/selfmap splits")

    ulf_root = Path(args.ulf_root)
    imports = _import_ulfloc(ulf_root)
    config = yaml.load(Path(args.config).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    original_source_path = Path(args.source_path)
    loader_source_path = resolve_ulfloc_source_path_for_loader(
        original_source_path,
        Path(args.output_dir) / "_ulf_loader_links",
    )
    args = argparse.Namespace(**vars(args))
    args.source_path = loader_source_path
    dataset = _dataset_namespace(args, config)
    train_images_to_read, train_list_path = resolve_train_images_to_read(
        Path(dataset.source_path),
        Path(args.train_list) if args.train_list is not None else None,
    )
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    dataset.eval = camera_split == "test"
    compute_device = str(args.compute_device)
    output_path = configure_sparse_feedback_eval_overrides(
        config,
        model_path=Path(dataset.model_path),
        input_log_dir=Path(args.input_log_dir) if args.input_log_dir is not None else None,
        scene_detector_checkpoint=Path(args.scene_detector_checkpoint)
        if args.scene_detector_checkpoint is not None
        else None,
        scene_detector_blend_alpha=float(args.scene_detector_blend_alpha),
        scene_detector_candidate_top_k=int(args.scene_detector_candidate_top_k),
        scene_detector_native_keep_fraction=float(args.scene_detector_native_keep_fraction),
    )

    if dataset.gaussian_type == "3dgs":
        gaussians = imports["GaussianModel"](dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = imports["GaussianModel_2dgs"](dataset.sh_degree)
    else:
        raise ValueError(f"unsupported gaussian_type: {dataset.gaussian_type}")

    scene_obj = imports["Scene"](
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        images_to_read=train_images_to_read,
        preload_cameras=True,
    )
    masks = None if args.no_masks else _load_masks(Path(dataset.source_path), dataset.images)
    ulfloc = imports["ULFLoc"](scene_obj, gaussians, masks, str(output_path), config)

    records: list[FeedbackMatchRecord] = []
    cross_view_matches: list[dict[str, Any]] = []
    pose_errors_t: list[float] = []
    pose_errors_r: list[float] = []
    processed_images = 0
    cameras = scene_obj.getTrainCameras() if camera_split == "train" else scene_obj.getTestCameras()
    for camera in cameras:
        if args.max_images and processed_images >= int(args.max_images):
            break
        gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        query_image = camera.original_image.to(compute_device)
        query_image = _apply_ulfloc_masks(query_image, masks, camera.image_name)
        sparse_result = ulfloc.loc_coarse(
            camera,
            query_image,
            camera.FoVx,
            camera.FoVy,
            return_matches=True,
        )
        pose_error_r_deg, pose_error_t_cm = imports["cal_pose_error"](sparse_result["pose_w2c"], gt_w2c)
        details = dict(sparse_result.get("matches", {}))
        details["pose_error_t_cm"] = float(pose_error_t_cm)
        details["pose_error_r_deg"] = float(pose_error_r_deg)
        details["selfmap_gt_w2c"] = gt_w2c
        details = compact_ulfloc_sparse_details(
            details,
            inliers_only=bool(args.inliers_only_for_bank),
            max_non_inlier_records=int(args.max_non_inlier_records_per_image),
        )
        records.extend(
            records_from_ulfloc_sparse_details(
                scene=str(args.scene),
                image_id=str(camera.image_name),
                details=details,
                split_name=split_name,
                source_role=str(args.source_role),
            )
        )
        cross_view_matches.extend(
            cross_view_matches_from_ulfloc_sparse_details(
                scene=str(args.scene),
                image_id=str(camera.image_name),
                details=details,
                split_name=split_name,
            )
        )
        pose_errors_t.append(float(pose_error_t_cm))
        pose_errors_r.append(float(pose_error_r_deg))
        processed_images += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_audit = {
        "audit_status": "passed",
        "source": f"ulfloc_scene_get{camera_split.capitalize()}Cameras",
        "split_name": split_name,
        "test_split_used": False,
        "camera_container": camera_split,
        "notes": (
            "ULF-Loc sparse feedback collected from the requested camera container with a non-test split name. "
            "If camera_container is test, it is intended only for train-dev validation overlays with an explicit train_list."
        ),
    }
    command = " ".join(sys.argv)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(args.scene),
        "split_name": split_name,
        "query_id_source": "ulfloc_train_image_id",
        "teacher_source": "ulfloc_sampled_landmarks",
        "feedback_role": "bootstrap_sparse_teacher",
        "paper_safe_role": "diagnostic_not_main_full_raw_feedback",
        "source_role": str(args.source_role),
        "main_feedback_source_required": "full_raw_gaussian_projection",
        "split_audit": split_audit,
        "command": command,
        "source_path": str(args.source_path),
        "original_source_path": str(original_source_path),
        "model_path": str(args.model_path),
        "config": str(args.config),
        "input_log_dir": None if args.input_log_dir is None else str(args.input_log_dir),
        "resolved_log_dir": str(output_path),
        "scene_detector_checkpoint": None
        if args.scene_detector_checkpoint is None
        else str(args.scene_detector_checkpoint),
        "scene_detector_enabled": args.scene_detector_checkpoint is not None,
        "scene_detector_blend_alpha": float(args.scene_detector_blend_alpha),
        "scene_detector_candidate_top_k": int(args.scene_detector_candidate_top_k),
        "scene_detector_native_keep_fraction": float(args.scene_detector_native_keep_fraction),
        "iteration": int(args.iteration),
        "data_device": str(args.data_device),
        "compute_device": compute_device,
        "max_images": int(args.max_images),
        "inliers_only_for_bank": bool(args.inliers_only_for_bank),
        "max_non_inlier_records_per_image": int(args.max_non_inlier_records_per_image),
        "train_list": train_list_path,
        "camera_split": camera_split,
        "images_to_read_count": len(train_images_to_read) if train_images_to_read is not None else None,
        "ulfloc_root": str(ulf_root),
    }
    bank_path = output_dir / "feedback_bank.jsonl"
    matches_path = output_dir / "matches.jsonl"
    save_feedback_bank(bank_path, records, metadata=metadata)
    save_cross_view_matches_jsonl(matches_path, cross_view_matches, metadata)
    audit = audit_feedback_bank_v3(bank_path)
    summary = summarize_feedback_bank(bank_path)
    summary.update(
        {
            "processed_images": int(processed_images),
            "cross_view_match_count": int(len(cross_view_matches)),
            "mean_sparse_te_cm": float(np.mean(pose_errors_t)) if pose_errors_t else None,
            "mean_sparse_re_deg": float(np.mean(pose_errors_r)) if pose_errors_r else None,
            "feedback_bank_audit_status": audit["audit_status"],
        }
    )
    manifest = dict(metadata)
    manifest.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "ulf_git_commit": _git_commit(ulf_root),
            "feedback_bank": str(bank_path),
            "cross_view_matches": str(matches_path),
            "metrics_summary": summary,
            "feedback_bank_audit": audit,
        }
    )
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics_summary.json", summary)
    _write_json(output_dir / "split_audit.json", split_audit)
    _write_json(output_dir / "feedback_bank_audit.json", audit)
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(_git_status(Path(__file__).resolve().parents[2]), encoding="utf-8")
    (output_dir / "ulf_git_status.txt").write_text(_git_status(ulf_root), encoding="utf-8")
    if audit["audit_status"] != "passed":
        raise RuntimeError(f"feedback bank audit failed: {audit['reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
