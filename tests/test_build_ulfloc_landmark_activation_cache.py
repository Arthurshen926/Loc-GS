import json
import pickle
import subprocess
import sys

import torch


def _write_ulfloc_log(path):
    path.mkdir()
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        dtype=torch.float32,
    )
    sampled_idx = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    with (path / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(features, handle)
    with (path / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)


def _write_impact(path, *, split_name="train_selfmap"):
    torch.save(
        {
            "schema_version": "solver_feedback_impact_v1",
            "split_name": split_name,
            "query_features": {
                "q1": torch.tensor([1.0, 0.0], dtype=torch.float32),
                "q2": torch.tensor([0.0, 1.0], dtype=torch.float32),
            },
            "detector_positive": {
                "q1": [{"gaussian_id": 10, "weight": 2.0}],
                "q2": [{"gaussian_id": 20, "weight": 1.0}],
            },
            "detector_negative": {
                "q1": [{"gaussian_id": 30, "weight": 3.0}],
                "q2": [{"gaussian_id": 40, "weight": 2.0}],
            },
            "landmark_positive": {
                10: {"support": 2.0},
                20: {"support": 1.0},
            },
            "landmark_negative": {
                30: {"risk": 3.0},
                40: {"risk": 2.0},
            },
            "split_audit": {"split_name": split_name, "test_split_used": split_name == "test"},
        },
        path,
    )


def _write_query_token_source(path, *, split_name="train_selfmap"):
    torch.save(
        {
            "schema_version": "sparse_solver_feedback_v4",
            "split_name": split_name,
            "correspondences": [
                {
                    "split_name": split_name,
                    "query_id": "q1",
                    "query_keypoint_index": 0,
                    "query_descriptor": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
                    "keypoint_xy": [32.0, 64.0],
                    "detector_score": 0.9,
                },
                {
                    "split_name": split_name,
                    "query_id": "q1",
                    "query_keypoint_index": 1,
                    "query_descriptor": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
                    "keypoint_xy": [96.0, 128.0],
                    "detector_score": 0.7,
                },
                {
                    "split_name": split_name,
                    "query_id": "q2",
                    "query_keypoint_index": 0,
                    "query_descriptor": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
                    "keypoint_xy": [16.0, 48.0],
                    "detector_score": 0.5,
                },
            ],
            "split_audit": {"split_name": split_name, "test_split_used": False, "official_test_used": False},
        },
        path,
    )


def _write_impact_without_query_features(path, *, split_name="train_selfmap"):
    torch.save(
        {
            "schema_version": "solver_feedback_impact_v1",
            "split_name": split_name,
            "detector_positive": {"q1": [{"gaussian_id": 10, "weight": 2.0}]},
            "detector_negative": {"q1": [{"gaussian_id": 30, "weight": 3.0}]},
            "landmark_positive": {10: {"support": 2.0}},
            "landmark_negative": {30: {"risk": 3.0}},
            "split_audit": {"split_name": split_name, "test_split_used": False},
        },
        path,
    )


def test_build_ulfloc_landmark_activation_cache_writes_cache_and_audit_bundle(tmp_path):
    source_log = tmp_path / "ulf_log"
    _write_ulfloc_log(source_log)
    impact = tmp_path / "impact.pt"
    _write_impact(impact)
    query_tokens = tmp_path / "query_tokens.pt"
    _write_query_token_source(query_tokens)
    output = tmp_path / "cache_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_landmark_activation_cache",
            "--impact_attribution",
            str(impact),
            "--query_token_source",
            str(query_tokens),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--top_n",
            "2",
            "--safe_core_min_support",
            "1.5",
            "--safe_core_max_risk",
            "0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    cache = torch.load(output / "cache.pt", map_location="cpu")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))

    assert cache["split_name"] == "train_selfmap"
    assert cache["query_dim"] == 3
    assert cache["landmark_dim"] == 3
    assert cache["landmark_ids"].tolist() == [10, 20, 30, 40]
    assert cache["safe_core_ids"].tolist() == [10]
    assert cache["examples"][0]["query_id"] == "q1"
    assert cache["examples"][0]["positive_ids"].tolist() == [0]
    assert cache["examples"][0]["negative_ids"].tolist() == [2]
    assert cache["query_token_dim"] == 6
    assert cache["landmark_token_dim"] == 7
    assert cache["landmark_tokens"].shape == (4, 7)
    assert cache["examples"][0]["query_tokens"].shape == (2, 6)
    assert cache["query_token_source_role"] == "per_keypoint_runtime_schema"
    assert "landmark_tokens" not in cache["examples"][0]
    assert cache["examples"][0]["solver_positive"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert cache["examples"][0]["harmful_negative"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert cache["examples"][0]["protected_support"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert manifest["scene"] == "ShopFacade"
    assert manifest["top_n"] == 2
    assert split_audit["test_split_used"] is False
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_build_ulfloc_landmark_activation_cache_rejects_test_split(tmp_path):
    source_log = tmp_path / "ulf_log"
    _write_ulfloc_log(source_log)
    impact = tmp_path / "impact.pt"
    _write_impact(impact, split_name="test")
    output = tmp_path / "cache_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_landmark_activation_cache",
            "--impact_attribution",
            str(impact),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_build_ulfloc_landmark_activation_cache_requires_query_token_source_by_default(tmp_path):
    source_log = tmp_path / "ulf_log"
    _write_ulfloc_log(source_log)
    impact = tmp_path / "impact.pt"
    _write_impact_without_query_features(impact)
    output = tmp_path / "cache_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_landmark_activation_cache",
            "--impact_attribution",
            str(impact),
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "query_token_source" in result.stderr


def test_build_ulfloc_landmark_activation_cache_allows_global_feature_diagnostic_only(tmp_path):
    source_log = tmp_path / "ulf_log"
    _write_ulfloc_log(source_log)
    impact = tmp_path / "impact.pt"
    _write_impact_without_query_features(impact)
    query_features = tmp_path / "query_features.pt"
    torch.save(
        {
            "split_name": "train_selfmap",
            "query_features": {"q1": torch.tensor([0.25, 0.75], dtype=torch.float32)},
            "split_audit": {"test_split_used": False},
        },
        query_features,
    )
    output = tmp_path / "cache_out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_landmark_activation_cache",
            "--impact_attribution",
            str(impact),
            "--query_features",
            str(query_features),
            "--allow_global_query_token_diagnostic",
            "--source_log_dir",
            str(source_log),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    cache = torch.load(output / "cache.pt", map_location="cpu")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert cache["query_dim"] == 2
    assert cache["examples"][0]["query_id"] == "q1"
    assert cache["examples"][0]["query_global_feature"].tolist() == [0.25, 0.75]
    assert cache["diagnostic_only"] is True
    assert cache["query_token_source_role"] == "global_feature_fallback"
    assert manifest["query_features"] == str(query_features)
