import json
from pathlib import Path

import numpy as np
import torch

from loc_gs.simulation.render_runner import (
    build_feature_map_cache_from_render_records,
    render_assets_from_manifest_records,
)
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
        "feature_map_path": str(tmp_path / "renders" / f"{query_id}.features.pt"),
    }


def _render_with_features(_record):
    descriptor_map = torch.zeros((2, 2, 3), dtype=torch.float32)
    descriptor_map[:, 0, 1] = torch.tensor([1.0, 0.0])
    descriptor_map[:, 1, 2] = torch.tensor([0.0, 1.0])
    return {
        "rgb": np.ones((2, 3, 3), dtype=np.float32) * 0.5,
        "depth": np.ones((2, 3), dtype=np.float32),
        "descriptor_map": descriptor_map,
        "score_map": torch.tensor([[0.1, 0.9, 0.2], [0.0, 0.3, 0.8]], dtype=torch.float32),
        "keypoints_yx": torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
        "keypoint_scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
    }


def test_render_runner_writes_feature_artifacts_and_aggregate_feature_map_cache(tmp_path: Path):
    summary, updated = render_assets_from_manifest_records(
        [_record(tmp_path)],
        scene="GreatCourt",
        split_name="train_selfmap",
        render_fn=_render_with_features,
    )

    feature_path = Path(updated[0]["feature_map_path"])
    assert summary["rendered_feature_map_count"] == 1
    assert summary["existing_feature_map_count"] == 1
    assert updated[0]["feature_map_exists"] is True
    assert feature_path.is_file()

    cache_path = tmp_path / "feature_map_cache.pt"
    cache_summary = build_feature_map_cache_from_render_records(
        updated,
        output_cache=cache_path,
        scene="GreatCourt",
        split_name="train_selfmap",
    )

    cache = torch.load(cache_path, map_location="cpu")
    assert cache_summary["schema_version"] == "internal_render_feature_map_cache_summary_v1"
    assert cache_summary["entry_count"] == 1
    assert cache_summary["missing_feature_map_count"] == 0
    assert cache["metadata"]["source"] == "internal_3dgs_render_feature_map_cache"
    assert cache["entries"][0]["image_id"] == "sim/GreatCourt/train_selfmap/000000"
    assert cache["entries"][0]["descriptor_map"].shape == (2, 2, 3)
    assert cache["entries"][0]["keypoints_yx"].tolist() == [[0.0, 1.0], [1.0, 2.0]]


def test_render_internal_3dgs_assets_cli_writes_feature_map_cache_when_features_exist(tmp_path: Path):
    manifest = tmp_path / "render_manifest.jsonl"
    feature_path = tmp_path / "renders" / "sim/GreatCourt/train_selfmap/000000.features.pt"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_id": "sim/GreatCourt/train_selfmap/000000",
            "descriptor_map": torch.ones((2, 2, 3), dtype=torch.float32),
            "score_map": torch.ones((2, 3), dtype=torch.float32),
        },
        feature_path,
    )
    record = _record(tmp_path)
    record["feature_map_exists"] = True
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
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
    cache = torch.load(out / "render_feature_map_cache.pt", map_location="cpu")
    assert summary["feature_map_cache_entry_count"] == 1
    assert summary["feature_map_cache"] == str(out / "render_feature_map_cache.pt")
    assert cache["entries"][0]["image_id"] == "sim/GreatCourt/train_selfmap/000000"
