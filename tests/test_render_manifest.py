import json
from pathlib import Path

import pytest

from loc_gs.simulation.render_manifest import build_render_manifest_from_plan_rows
from loc_gs.scripts.build_internal_render_manifest import main


def _plan_rows():
    return [
        {
            "schema_version": "internal_3dgs_simulation_plan_summary_v1",
            "scene": "GreatCourt",
            "split_name": "train_selfmap",
            "synthetic_query_id": "sim/GreatCourt/train_selfmap/000000",
            "source_image_id": "a.png",
            "render_pose_c2w": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "render_intrinsics": {"width": 100, "height": 80, "fx": 50.0, "fy": 50.0, "cx": 49.5, "cy": 39.5},
            "render_contract": "posed_3dgs_camera_v1",
            "render_engine": "3dgs",
        },
        {
            "scene": "GreatCourt",
            "split_name": "train_selfmap",
            "synthetic_query_id": "sim/GreatCourt/train_selfmap/000001",
            "source_image_id": "b.png",
            "render_pose_c2w": [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "render_intrinsics": {"width": 100, "height": 80, "fx": 50.0, "fy": 50.0, "cx": 49.5, "cy": 39.5},
            "render_contract": "posed_3dgs_camera_v1",
            "render_engine": "3dgs",
        },
    ]


def test_render_manifest_maps_simulation_rows_to_expected_assets(tmp_path: Path):
    rgb = tmp_path / "renders" / "sim" / "GreatCourt" / "train_selfmap" / "000000.png"
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b"fake")

    summary, records = build_render_manifest_from_plan_rows(
        _plan_rows(),
        scene="GreatCourt",
        split_name="train_selfmap",
        render_root=tmp_path / "renders",
    )

    assert summary["schema_version"] == "internal_render_manifest_summary_v1"
    assert summary["render_request_count"] == 2
    assert summary["rendered_rgb_count"] == 1
    assert summary["missing_rgb_count"] == 1
    assert records[0]["rgb_exists"] is True
    assert records[0]["rgb_path"] == str(rgb)
    assert records[0]["render_pose_c2w"][0] == [1.0, 0.0, 0.0, 0.0]
    assert records[1]["rgb_exists"] is False
    assert records[1]["role"] == "training_render_asset"


def test_render_manifest_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_render_manifest_from_plan_rows(
            _plan_rows(),
            scene="GreatCourt",
            split_name="test",
            render_root="renders",
        )


def test_render_manifest_cli_writes_auditable_bundle(tmp_path: Path):
    plan = tmp_path / "simulation_plan.jsonl"
    plan.write_text("\n".join(json.dumps(row) for row in _plan_rows()) + "\n", encoding="utf-8")
    rgb = tmp_path / "renders" / "sim" / "GreatCourt" / "train_selfmap" / "000000.png"
    rgb.parent.mkdir(parents=True)
    rgb.write_bytes(b"fake")
    out = tmp_path / "manifest"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_selfmap",
            "--simulation_plan",
            str(plan),
            "--render_root",
            str(tmp_path / "renders"),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out / "render_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["render_request_count"] == 2
    assert summary["rendered_rgb_count"] == 1
    assert summary["missing_rgb_count"] == 1
    assert manifest["schema_version"] == "internal_render_manifest_manifest_v1"
    assert manifest["dense_inference_enabled"] is False
    assert rows[0]["synthetic_query_id"] == "sim/GreatCourt/train_selfmap/000000"
    assert (out / "missing_render_ids.txt").read_text(encoding="utf-8").splitlines() == [
        "sim/GreatCourt/train_selfmap/000001"
    ]
