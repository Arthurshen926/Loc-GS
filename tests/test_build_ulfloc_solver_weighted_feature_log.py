import json
import pickle
import subprocess
import sys

import torch
import torch.nn.functional as F


def test_build_ulfloc_solver_weighted_feature_log_replaces_only_feature_pickle(tmp_path):
    source = tmp_path / "source_log"
    source.mkdir()
    source_features = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    sampled_idx = torch.tensor([5, 9], dtype=torch.long)
    with (source / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(source_features, handle)
    with (source / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)
    (source / "config.yaml").write_text("sample: {}\n", encoding="utf-8")

    artifact = tmp_path / "selected_landmark_descriptors.pt"
    fused = F.normalize(torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32), p=2, dim=-1)
    torch.save(
        {
            "descriptors": fused,
            "base_descriptors": source_features,
            "gaussian_ids": sampled_idx,
            "metadata": {
                "descriptor_mode": "solver_weighted_pair_cache_fusion",
                "source_split_name": "train_dev_seed13",
            },
        },
        artifact,
    )
    output = tmp_path / "fused_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_weighted_feature_log",
            "--source_log_dir",
            str(source),
            "--descriptor_artifact",
            str(artifact),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--source_feature_min_cosine",
            "0.99",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    with (output / "keypoints_sampled_idx.pkl").open("rb") as handle:
        saved_idx = pickle.load(handle)
    manifest = json.loads((output / "solver_weighted_feature_log_manifest.json").read_text())
    split_audit = json.loads((output / "split_audit.json").read_text())

    assert torch.allclose(torch.as_tensor(saved).cpu(), fused)
    assert torch.as_tensor(saved_idx).tolist() == [5, 9]
    assert manifest["artifact_metadata"]["source_split_name"] == "train_dev_seed13"
    assert manifest["same_sparse_geometry"] is True
    assert split_audit["test_split_used"] is False
    assert split_audit["official_test_used"] is False


def test_build_ulfloc_solver_weighted_feature_log_allows_matching_zero_fallback_rows(tmp_path):
    source = tmp_path / "source_log"
    source.mkdir()
    source_features = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    sampled_idx = torch.tensor([5, 9], dtype=torch.long)
    with (source / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(source_features, handle)
    with (source / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)

    artifact = tmp_path / "selected_landmark_descriptors.pt"
    torch.save(
        {
            "descriptors": source_features,
            "base_descriptors": source_features,
            "gaussian_ids": sampled_idx,
            "metadata": {"source_split_name": "train_dev_seed13"},
        },
        artifact,
    )
    output = tmp_path / "fused_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_weighted_feature_log",
            "--source_log_dir",
            str(source),
            "--descriptor_artifact",
            str(artifact),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--source_feature_min_cosine",
            "0.99",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "solver_weighted_feature_log_manifest.json").read_text())
    assert manifest["source_feature_match"]["matching_zero_count"] == 1
