from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

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
            )
        )
    return specs


def summarize_simulation_plan(specs: Sequence[SimulatedQuerySpec]) -> dict[str, object]:
    return {
        "schema_version": "internal_3dgs_simulation_plan_summary_v1",
        "sample_count": int(len(specs)),
        "source_image_count": int(len({spec.source_image_id for spec in specs})),
        "render_engine": "3dgs",
    }
