import json
from pathlib import Path

import numpy as np
import pytest

from loc_gs.core.camera import CameraIntrinsics, CameraRecord
from loc_gs.scripts.build_internal_simulation_plan import main
from loc_gs.simulation.query_sampler import SimulationSamplerConfig, sample_simulated_queries


def _records():
    return {
        "a.png": CameraRecord("a.png", CameraIntrinsics(width=100, height=80, fx=50.0, fy=50.0, cx=50.0, cy=40.0)),
        "b.png": CameraRecord("b.png", CameraIntrinsics(width=100, height=80, fx=50.0, fy=50.0, cx=50.0, cy=40.0)),
    }


def test_simulation_sampler_is_seeded_and_rejects_test_split():
    cfg = SimulationSamplerConfig(sample_count=3, seed=7, translation_std_m=0.1, yaw_std_deg=2.0)

    first = sample_simulated_queries(_records(), scene="GreatCourt", split_name="train", cfg=cfg)
    second = sample_simulated_queries(_records(), scene="GreatCourt", split_name="train", cfg=cfg)

    assert [item.to_json_dict() for item in first] == [item.to_json_dict() for item in second]
    assert len(first) == 3
    assert first[0].source_image_id in {"a.png", "b.png"}
    assert first[0].synthetic_query_id.startswith("sim/GreatCourt/train/")
    with pytest.raises(ValueError, match="test split"):
        sample_simulated_queries(_records(), scene="GreatCourt", split_name="test", cfg=cfg)


def test_simulation_sampler_exports_render_pose_contract_for_3dgs():
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    records = {
        "posed.png": CameraRecord(
            "posed.png",
            CameraIntrinsics(width=120, height=90, fx=60.0, fy=61.0, cx=59.5, cy=44.5),
            pose_c2w=pose,
        )
    }
    cfg = SimulationSamplerConfig(
        sample_count=1,
        seed=1,
        translation_std_m=0.0,
        yaw_std_deg=0.0,
        pitch_std_deg=0.0,
        roll_std_deg=0.0,
    )

    spec = sample_simulated_queries(records, scene="GreatCourt", split_name="train", cfg=cfg)[0]
    row = spec.to_json_dict()

    np.testing.assert_allclose(row["source_pose_c2w"], pose)
    np.testing.assert_allclose(row["render_pose_c2w"], pose)
    assert row["render_intrinsics"] == {
        "width": 120,
        "height": 90,
        "fx": 60.0,
        "fy": 61.0,
        "cx": 59.5,
        "cy": 44.5,
    }
    assert row["render_contract"] == "posed_3dgs_camera_v1"


def test_simulation_plan_cli_writes_audited_plan(tmp_path: Path):
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
        json.dumps(
            [
                {"img_name": "a.png", "width": 100, "height": 80, "fx": 50.0, "fy": 50.0},
                {"img_name": "b.png", "width": 100, "height": 80, "fx": 50.0, "fy": 50.0},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "sim"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_selfmap",
            "--cameras_json",
            str(cameras),
            "--output_dir",
            str(out),
            "--sample_count",
            "4",
            "--seed",
            "11",
        ]
    )

    assert rc == 0
    plan_rows = [json.loads(line) for line in (out / "simulation_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(plan_rows) == 4
    assert summary["sample_count"] == 4
    assert split_audit["audit_status"] == "passed"
    assert manifest["inference_stage"] == "simulation_plan_generation"
