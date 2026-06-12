from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from loc_gs.core.camera import CameraRecord
from loc_gs.sparse.audit import reject_test_split


@dataclass(frozen=True)
class SimulationSamplerConfig:
    sample_count: int
    seed: int = 13
    translation_std_m: float = 0.25
    yaw_std_deg: float = 5.0
    pitch_std_deg: float = 2.0
    roll_std_deg: float = 2.0


@dataclass(frozen=True)
class SimulatedQuerySpec:
    scene: str
    split_name: str
    synthetic_query_id: str
    source_image_id: str
    translation_delta_m: tuple[float, float, float]
    rotation_delta_deg: tuple[float, float, float]
    source_pose_c2w: tuple[tuple[float, float, float, float], ...] | None = None
    render_pose_c2w: tuple[tuple[float, float, float, float], ...] | None = None
    render_intrinsics: dict[str, int | float] | None = None
    render_contract: str = "posed_3dgs_camera_v1"
    render_engine: str = "3dgs"
    role: str = "training_simulation"

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_simulated_queries(
    camera_records: Mapping[str, CameraRecord],
    *,
    scene: str,
    split_name: str,
    cfg: SimulationSamplerConfig,
) -> list[SimulatedQuerySpec]:
    split = reject_test_split(split_name, purpose="3DGS simulation sample generation")
    image_ids = sorted(camera_records.keys())
    if not image_ids:
        raise ValueError("at least one camera record is required for simulation sampling")
    rng = random.Random(int(cfg.seed))
    specs: list[SimulatedQuerySpec] = []
    for idx in range(int(cfg.sample_count)):
        image_id = image_ids[rng.randrange(len(image_ids))]
        translation = tuple(round(rng.gauss(0.0, float(cfg.translation_std_m)), 6) for _ in range(3))
        rotation = (
            round(rng.gauss(0.0, float(cfg.yaw_std_deg)), 6),
            round(rng.gauss(0.0, float(cfg.pitch_std_deg)), 6),
            round(rng.gauss(0.0, float(cfg.roll_std_deg)), 6),
        )
        specs.append(
            SimulatedQuerySpec(
                scene=str(scene),
                split_name=split,
                synthetic_query_id=f"sim/{scene}/{split}/{idx:06d}",
                source_image_id=image_id,
                translation_delta_m=translation,
                rotation_delta_deg=rotation,
                source_pose_c2w=_pose_to_tuple(camera_records[image_id].pose_c2w),
                render_pose_c2w=_render_pose_c2w(camera_records[image_id], translation, rotation),
                render_intrinsics=_intrinsics_to_json(camera_records[image_id]),
            )
        )
    return specs


def summarize_simulation_plan(specs: Sequence[SimulatedQuerySpec]) -> dict[str, object]:
    return {
        "schema_version": "internal_3dgs_simulation_plan_summary_v1",
        "sample_count": int(len(specs)),
        "source_image_count": int(len({spec.source_image_id for spec in specs})),
        "posed_render_contract_count": int(sum(1 for spec in specs if spec.render_pose_c2w is not None)),
        "render_engine": "3dgs",
    }


def _intrinsics_to_json(record: CameraRecord) -> dict[str, int | float]:
    intr = record.intrinsics
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
    }


def _pose_to_tuple(pose: np.ndarray | None) -> tuple[tuple[float, float, float, float], ...] | None:
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return tuple(tuple(float(value) for value in row) for row in arr)


def _render_pose_c2w(
    record: CameraRecord,
    translation_delta_m: Sequence[float],
    rotation_delta_deg: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...] | None:
    if record.pose_c2w is None:
        return None
    pose = np.asarray(record.pose_c2w, dtype=np.float64).reshape(4, 4).copy()
    pose[:3, :3] = pose[:3, :3] @ _rotation_delta_matrix(rotation_delta_deg)
    pose[:3, 3] = pose[:3, 3] + np.asarray(translation_delta_m, dtype=np.float64).reshape(3)
    return _pose_to_tuple(pose)


def _rotation_delta_matrix(rotation_delta_deg: Sequence[float]) -> np.ndarray:
    yaw, pitch, roll = [np.deg2rad(float(value)) for value in rotation_delta_deg]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx
