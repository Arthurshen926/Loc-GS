from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class GaussianRenderRequest:
    scene: str
    split_name: str
    synthetic_query_id: str
    render_engine: str
    render_contract: str
    pose_c2w: np.ndarray
    intrinsics: CameraIntrinsics
    rgb_path: str | None = None
    depth_path: str | None = None
    feature_map_path: str | None = None

    @property
    def pose_w2c(self) -> np.ndarray:
        return np.linalg.inv(np.asarray(self.pose_c2w, dtype=np.float64).reshape(4, 4))

    @classmethod
    def from_manifest_record(cls, record: Mapping[str, Any]) -> "GaussianRenderRequest":
        split = reject_test_split(str(record.get("split_name") or "unknown"), purpose="internal 3DGS render request")
        engine = str(record.get("render_engine") or "3dgs")
        contract = str(record.get("render_contract") or "posed_3dgs_camera_v1")
        if engine != "3dgs":
            raise ValueError(f"unsupported Gaussian render engine: {engine}")
        if contract != "posed_3dgs_camera_v1":
            raise ValueError(f"unsupported Gaussian render contract: {contract}")
        intr = record.get("render_intrinsics")
        if not isinstance(intr, Mapping):
            raise ValueError("render manifest record is missing render_intrinsics")
        return cls(
            scene=str(record.get("scene") or "unknown"),
            split_name=split,
            synthetic_query_id=str(record.get("synthetic_query_id") or record.get("image_id") or ""),
            render_engine=engine,
            render_contract=contract,
            pose_c2w=np.asarray(record.get("render_pose_c2w"), dtype=np.float64).reshape(4, 4),
            intrinsics=CameraIntrinsics(
                width=int(intr["width"]),
                height=int(intr["height"]),
                fx=float(intr["fx"]),
                fy=float(intr["fy"]),
                cx=float(intr["cx"]),
                cy=float(intr["cy"]),
            ),
            rgb_path=None if not record.get("rgb_path") else str(record["rgb_path"]),
            depth_path=None if not record.get("depth_path") else str(record["depth_path"]),
            feature_map_path=None if not record.get("feature_map_path") else str(record["feature_map_path"]),
        )


def render_internal_3dgs_record(
    record: Mapping[str, Any],
    *,
    gaussian_ply: str | Path,
    feature_checkpoint: str | Path | None = None,
    device: str = "cuda",
    latent_dim: int = 16,
    render_feature_map: bool = True,
) -> dict[str, np.ndarray]:
    from loc_gs.simulation.render_runner import render_record_with_internal_3dgs

    GaussianRenderRequest.from_manifest_record(record)
    return render_record_with_internal_3dgs(
        record,
        gaussian_ply=gaussian_ply,
        feature_checkpoint=feature_checkpoint,
        device=str(device),
        latent_dim=int(latent_dim),
        render_feature_map=bool(render_feature_map),
    )
