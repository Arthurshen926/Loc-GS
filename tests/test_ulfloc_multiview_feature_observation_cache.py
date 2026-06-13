import json
import pickle
import subprocess
import sys

import torch

from loc_gs.training.ulfloc_multiview_feature_observations import build_multiview_feature_observation_cache
from loc_gs.scripts.build_ulfloc_multiview_feature_observation_cache import (
    _target_view_gaussian_ids_from_impact,
)


def test_build_multiview_feature_observation_cache_uses_original_view_descriptors():
    sampled_idx = torch.tensor([10, 20], dtype=torch.long)
    observations = {
        "im1.png": [
            {
                "gaussian_id": 10,
                "descriptor": [1.0, 0.0],
                "geometry_weight": 0.7,
                "keypoint_xy": [20.0, 10.0],
                "depth_m": 3.0,
            },
            {
                "gaussian_id": 30,
                "descriptor": [0.0, 1.0],
                "geometry_weight": 0.5,
            },
        ],
    }

    cache, metrics = build_multiview_feature_observation_cache(
        observations,
        sampled_idx=sampled_idx,
        scene="ShopFacade",
        split_name="train_selfmap",
    )

    assert cache["schema_version"] == "ulfloc_multiview_feature_observation_cache_v1"
    assert cache["observation_descriptor_source"] == "ulf_original_multiview_feature_observation"
    assert cache["observation_id_space"] == "gaussian_id"
    assert set(cache["observations"]) == {10}
    row = cache["observations"][10][0]
    assert row["source_view_id"] == "im1.png"
    assert row["positive_weight"] == 0.7
    assert row["descriptor"].tolist() == [1.0, 0.0]
    assert metrics["input_observation_count"] == 2
    assert metrics["kept_observation_count"] == 1
    assert metrics["not_sampled_observation_count"] == 1


def test_build_ulfloc_multiview_feature_observation_cache_cli_writes_audit_bundle(tmp_path):
    source_log = tmp_path / "log"
    source_log.mkdir()
    sampled_idx = torch.tensor([10, 20], dtype=torch.long)
    with (source_log / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_ulf_multiview_feature_observations_v1",
                "scene": "ShopFacade",
                "split_name": "train_selfmap",
                "observations": {
                    "im1.png": [
                        {"gaussian_id": 10, "descriptor": [1.0, 0.0], "geometry_weight": 0.8},
                    ],
                    "im2.png": [
                        {"gaussian_id": 20, "descriptor": [0.0, 1.0], "geometry_weight": 0.6},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_multiview_feature_observation_cache",
            "--input_observations_json",
            str(observations_path),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_selfmap",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    cache = torch.load(output / "observation_cache.pt", map_location="cpu")
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))
    assert torch.equal(cache["sampled_idx"], sampled_idx)
    assert cache["observation_descriptor_source"] == "ulf_original_multiview_feature_observation"
    assert metrics["kept_observation_count"] == 2
    assert split_audit["test_split_used"] is False
    assert (output / "manifest.json").exists()
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_build_ulfloc_multiview_feature_observation_cache_cli_rejects_test_split(tmp_path):
    source_log = tmp_path / "log"
    source_log.mkdir()
    with (source_log / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([10], dtype=torch.long), handle)
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps({"split_name": "train_selfmap", "observations": {}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_multiview_feature_observation_cache",
            "--input_observations_json",
            str(observations_path),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(tmp_path / "out"),
            "--scene",
            "ShopFacade",
            "--split_name",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_impact_filter_extracts_exact_view_gaussian_pairs_and_rejects_test():
    impact = {
        "split_name": "train_selfmap",
        "view_positive": {
            ("10", "seq/frame0001.png"): {"weight": 2.0},
            "20|seq/frame0002.png": {"weight": 1.0},
        },
        "view_negative": {
            ("30", "seq/frame0001.png"): {"weight": 1.0},
        },
    }

    targets = _target_view_gaussian_ids_from_impact(impact)

    assert targets == {
        "seq/frame0001.png": {10, 30},
        "seq/frame0002.png": {20},
    }

    bad = {"split_name": "test", "view_positive": {("10", "x.png"): {"weight": 1.0}}}
    try:
        _target_view_gaussian_ids_from_impact(bad)
    except ValueError as exc:
        assert "test split" in str(exc)
    else:
        raise AssertionError("expected test split impact to be rejected")
