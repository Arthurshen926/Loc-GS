from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from loc_gs.data.superpoint_cache import superpoint_score_map_from_logits
from loc_gs.sparse.audit import reject_test_split
from loc_gs.sparse.query_feature_cache import _sample_descriptors_bilinear, _select_feature_map_keypoints


@torch.no_grad()
def extract_superpoint_feature_dir_for_queries(
    *,
    scene: str,
    split_name: str,
    data_root: str | Path,
    query_ids: Sequence[str],
    output_dir: str | Path,
    model: torch.nn.Module,
    device: torch.device | str,
    batch_size: int = 16,
    strict_missing: bool = True,
    amp: bool = False,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal SuperPoint query feature extraction")
    root = Path(data_root)
    out = Path(output_dir)
    queries = [str(query_id) for query_id in query_ids]
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for query_id in queries:
        path = _resolve_query_path(root, query_id)
        if path.exists():
            resolved.append((query_id, path))
        else:
            missing.append(query_id)
    if missing and strict_missing:
        raise FileNotFoundError(f"missing query images under {root}: {missing[:10]}")

    for subdir in ("descriptor", "detector"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    target_h = 0
    target_w = 0
    if resolved:
        target_h, target_w = _target_size_for_superpoint(resolved[0][1])
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    frame_manifest: list[dict[str, object]] = []
    descriptor_dim = 0
    detector_dim = 0
    feature_h = 0
    feature_w = 0
    for start in range(0, len(resolved), int(batch_size)):
        batch = resolved[start : start + int(batch_size)]
        images = torch.stack(
            [_load_query_image(path, target_h=target_h, target_w=target_w) for _, path in batch],
            dim=0,
        ).to(device_obj)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device_obj.type == "cuda"):
            descriptors, detector_logits = model(images)
        descriptors = descriptors.detach().float().cpu()
        detector_logits = detector_logits.detach().float().cpu()
        if descriptors.ndim != 4:
            raise ValueError("SuperPoint model descriptors must have shape [B,C,H,W]")
        if detector_logits.ndim != 4 or int(detector_logits.shape[1]) != 65:
            raise ValueError("SuperPoint model detector logits must have shape [B,65,H,W]")
        descriptor_dim = int(descriptors.shape[1])
        detector_dim = int(detector_logits.shape[1])
        feature_h = int(descriptors.shape[2])
        feature_w = int(descriptors.shape[3])
        for local_idx, (query_id, path) in enumerate(batch):
            saved_stem = _saved_stem(query_id)
            torch.save(descriptors[local_idx].half(), out / "descriptor" / f"{saved_stem}.pt")
            torch.save(detector_logits[local_idx].half(), out / "detector" / f"{saved_stem}.pt")
            frame_manifest.append(
                {
                    "source_rank": int(start + local_idx),
                    "source_file": query_id,
                    "source_path": str(path),
                    "saved_stem": saved_stem,
                }
            )

    frame_payload = {
        "schema_version": "internal_superpoint_feature_dir_v1",
        "scene": str(scene),
        "split_name": split,
        "data_root": str(root),
        "num_frames": int(len(frame_manifest)),
        "feature_type": "superpoint",
        "descriptor_dim": int(descriptor_dim),
        "detector_dim": int(detector_dim),
        "target_height": int(target_h),
        "target_width": int(target_w),
        "feature_height": int(feature_h),
        "feature_width": int(feature_w),
        "frames": frame_manifest,
    }
    (out / "frame_manifest.json").write_text(
        json.dumps(frame_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "internal_superpoint_query_feature_extraction_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "data_root": str(root),
        "output_dir": str(out),
        "requested_query_count": int(len(queries)),
        "extracted_query_count": int(len(frame_manifest)),
        "missing_query_count": int(len(missing)),
        "missing_query_ids_preview": missing[:10],
        "descriptor_dim": int(descriptor_dim),
        "detector_dim": int(detector_dim),
        "target_height": int(target_h),
        "target_width": int(target_w),
        "feature_height": int(feature_h),
        "feature_width": int(feature_w),
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


@torch.no_grad()
def extract_superpoint_query_feature_cache_for_queries(
    *,
    scene: str,
    split_name: str,
    data_root: str | Path,
    query_ids: Sequence[str],
    output_cache: str | Path,
    model: torch.nn.Module,
    device: torch.device | str,
    max_keypoints: int = 2048,
    score_threshold: float | None = None,
    batch_size: int = 16,
    strict_missing: bool = True,
    amp: bool = False,
) -> dict[str, object]:
    split = reject_test_split(split_name, purpose="internal SuperPoint image query feature cache")
    root = Path(data_root)
    queries = [str(query_id) for query_id in query_ids]
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    max_k = int(max_keypoints)
    if max_k <= 0:
        raise ValueError("max_keypoints must be positive")
    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for query_id in queries:
        path = _resolve_query_path(root, query_id)
        if path.exists():
            resolved.append((query_id, path))
        else:
            missing.append(query_id)
    if missing and strict_missing:
        raise FileNotFoundError(f"missing query images under {root}: {missing[:10]}")

    target_h = 0
    target_w = 0
    if resolved:
        target_h, target_w = _target_size_for_superpoint(resolved[0][1])
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()
    rows: list[dict[str, object]] = []
    descriptor_dim = 0
    feature_h = 0
    feature_w = 0
    for start in range(0, len(resolved), int(batch_size)):
        batch = resolved[start : start + int(batch_size)]
        images = torch.stack(
            [_load_query_image(path, target_h=target_h, target_w=target_w) for _, path in batch],
            dim=0,
        ).to(device_obj)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device_obj.type == "cuda"):
            descriptors, detector_logits = model(images)
        descriptors = descriptors.detach().float().cpu()
        detector_logits = detector_logits.detach().float().cpu()
        if descriptors.ndim != 4:
            raise ValueError("SuperPoint model descriptors must have shape [B,C,H,W]")
        if detector_logits.ndim != 4 or int(detector_logits.shape[1]) != 65:
            raise ValueError("SuperPoint model detector logits must have shape [B,65,H,W]")
        descriptor_dim = int(descriptors.shape[1])
        feature_h = int(descriptors.shape[2])
        feature_w = int(descriptors.shape[3])
        for local_idx, (query_id, _) in enumerate(batch):
            descriptor_map = descriptors[local_idx]
            score_map = torch.as_tensor(
                superpoint_score_map_from_logits(detector_logits[local_idx]),
                dtype=torch.float32,
            )
            keypoints_yx, keypoint_scores = _select_feature_map_keypoints(
                entry={"score_map": score_map},
                max_keypoints=max_k,
                score_threshold=score_threshold,
            )
            if keypoints_yx.numel() == 0:
                continue
            query_desc = _sample_descriptors_bilinear(descriptor_map, keypoints_yx / 8.0)
            query_desc = F.normalize(query_desc, p=2, dim=1)
            for row_idx in range(int(keypoints_yx.shape[0])):
                rows.append(
                    {
                        "image_id": query_id,
                        "keypoint_id": f"kp_{row_idx:06d}",
                        "query_yx": keypoints_yx[row_idx].cpu(),
                        "query_desc": query_desc[row_idx].cpu(),
                        "query_score": float(keypoint_scores[row_idx]),
                    }
                )

    covered_queries = _dedupe(row["image_id"] for row in rows)
    missing_or_empty_queries = [query_id for query_id in queries if query_id not in set(covered_queries)]
    query_yx = (
        torch.stack([torch.as_tensor(row["query_yx"], dtype=torch.float32) for row in rows], dim=0)
        if rows
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    query_desc = (
        torch.stack([torch.as_tensor(row["query_desc"], dtype=torch.float32) for row in rows], dim=0)
        if rows
        else torch.zeros((0, int(descriptor_dim)), dtype=torch.float32)
    )
    query_score = (
        torch.tensor([float(row["query_score"]) for row in rows], dtype=torch.float32)
        if rows
        else torch.zeros((0,), dtype=torch.float32)
    )
    split_audit = {
        "schema_version": "internal_split_audit_v1",
        "audit_status": "passed",
        "split_name": split,
        "official_test_used": False,
        "test_split_used": False,
    }
    output = {
        "metadata": {
            "schema_version": "internal_query_feature_cache_v1",
            "scene": str(scene),
            "split_name": split,
            "source": "internal_superpoint_image_sampler",
            "data_root": str(root),
            "split_audit": dict(split_audit),
        },
        "image_id": [str(row["image_id"]) for row in rows],
        "keypoint_id": [str(row["keypoint_id"]) for row in rows],
        "query_yx": query_yx.float(),
        "query_desc": query_desc.float(),
        "query_score": query_score.float(),
    }
    output_path = Path(output_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return {
        "schema_version": "internal_query_feature_cache_summary_v1",
        "scene": str(scene),
        "split_name": split,
        "source": "internal_superpoint_image_sampler",
        "data_root": str(root),
        "output_cache": str(output_path),
        "requested_query_count": int(len(queries)),
        "query_count": int(len(covered_queries)),
        "keypoint_count": int(len(rows)),
        "missing_query_count": int(len(missing_or_empty_queries)),
        "missing_query_ids_preview": missing_or_empty_queries[:10],
        "descriptor_dim": int(descriptor_dim),
        "max_keypoints": int(max_k),
        "score_threshold": None if score_threshold is None else float(score_threshold),
        "target_height": int(target_h),
        "target_width": int(target_w),
        "feature_height": int(feature_h),
        "feature_width": int(feature_w),
        "dense_teacher_enabled": False,
        "dense_inference_enabled": False,
        "external_runtime_dependency": "forbidden",
    }


def _resolve_query_path(data_root: Path, query_id: str) -> Path:
    candidate = (data_root / query_id).resolve()
    root = data_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"query id escapes data_root: {query_id}") from exc
    return candidate


def _target_size_for_superpoint(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    target_h = (int(height) // 8) * 8
    target_w = (int(width) // 8) * 8
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"image is too small for SuperPoint stride-8 extraction: {path}")
    return target_h, target_w


def _load_query_image(path: Path, *, target_h: int, target_w: int) -> torch.Tensor:
    with Image.open(path) as image:
        gray = image.convert("L")
        array = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    if int(tensor.shape[-2]) != int(target_h) or int(tensor.shape[-1]) != int(target_w):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(int(target_h), int(target_w)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.float()


def _saved_stem(query_id: str) -> str:
    clean = Path(query_id).with_suffix("").as_posix().replace("/", "__")
    return clean.replace("\\", "__")


def _dedupe(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = str(raw_value)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
