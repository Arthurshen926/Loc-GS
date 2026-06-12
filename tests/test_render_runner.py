import json
from pathlib import Path

import numpy as np
import pytest

from loc_gs.simulation.render_runner import render_assets_from_manifest_records
from loc_gs.scripts.render_internal_3dgs_assets import main


def _record(tmp_path: Path, *, query_id: str = "sim/GreatCourt/train_selfmap/000000") -> dict[str, object]:
    return {
        "scene": "GreatCourt",
        "split_name": "train_selfmap",
        "synthetic_query_id": query_id,
        "render_pose_c2w": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "render_intrinsics": {"width": 3, "height": 2, "fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 0.5},
        "rgb_path": str(tmp_path / "renders" / f"{query_id}.png"),
        "depth_path": str(tmp_path / "renders" / f"{query_id}.depth.npy"),
    }


def test_render_assets_from_manifest_records_writes_rgb_depth_and_updated_manifest(tmp_path: Path):
    def fake_renderer(_record):
        return {
            "rgb": np.ones((2, 3, 3), dtype=np.float32) * 0.5,
            "depth": np.ones((2, 3), dtype=np.float32) * 7.0,
        }

    summary, updated = render_assets_from_manifest_records(
        [_record(tmp_path)],
        scene="GreatCourt",
        split_name="train_selfmap",
        render_fn=fake_renderer,
    )

    rgb_path = Path(updated[0]["rgb_path"])
    depth_path = Path(updated[0]["depth_path"])
    assert summary["schema_version"] == "internal_render_runner_summary_v1"
    assert summary["render_request_count"] == 1
    assert summary["rendered_rgb_count"] == 1
    assert summary["failed_render_count"] == 0
    assert rgb_path.is_file()
    assert depth_path.is_file()
    assert np.load(depth_path).shape == (2, 3)
    assert updated[0]["rgb_exists"] is True
    assert updated[0]["depth_exists"] is True
    assert updated[0]["render_status"] == "rendered"


def test_render_assets_from_manifest_records_rejects_test_split(tmp_path: Path):
    with pytest.raises(ValueError, match="test split"):
        render_assets_from_manifest_records(
            [_record(tmp_path)],
            scene="GreatCourt",
            split_name="test",
            render_fn=lambda _record: {"rgb": np.zeros((1, 1, 3), dtype=np.float32)},
        )


def test_render_internal_3dgs_assets_cli_dry_run_writes_auditable_bundle(tmp_path: Path):
    manifest = tmp_path / "render_manifest.jsonl"
    manifest.write_text(json.dumps(_record(tmp_path)) + "\n", encoding="utf-8")
    out = tmp_path / "out"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_selfmap",
            "--render_manifest",
            str(manifest),
            "--output_dir",
            str(out),
            "--dry_run",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out / "render_manifest.updated.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["dry_run"] is True
    assert summary["render_request_count"] == 1
    assert summary["rendered_rgb_count"] == 0
    assert run_manifest["schema_version"] == "internal_render_runner_manifest_v1"
    assert run_manifest["external_runtime_dependency"] == "forbidden"
    assert rows[0]["render_status"] == "dry_run"
    assert (out / "command.txt").is_file()
