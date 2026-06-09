import subprocess
import sys
import json

import torch
import torch.nn.functional as F


def test_build_solver_weighted_feature_fusion_cli_writes_artifact(tmp_path):
    pair_cache = tmp_path / "pairs.pt"
    output = tmp_path / "selected_landmark_descriptors.pt"
    torch.save(
        {
            "base_landmark_desc": F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1),
            "base_gaussian_id": torch.tensor([5, 9], dtype=torch.long),
            "query_desc": F.normalize(torch.tensor([[0.0, 1.0]], dtype=torch.float32), p=2, dim=-1),
            "landmark_id": torch.tensor([[0, 1]], dtype=torch.long),
            "candidate_mask": torch.tensor([[True, True]]),
            "cosine": torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            "reprojection_error": torch.tensor([[1.0, 9.0]], dtype=torch.float32),
            "metadata": {"source_split_name": "train", "feedback_bank_split_name": "selfmap_train"},
        },
        pair_cache,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_weighted_feature_fusion",
            "--pair_cache",
            str(pair_cache),
            "--output",
            str(output),
            "--trust_alpha",
            "0.5",
            "--reprojection_threshold_px",
            "3.0",
            "--min_cosine",
            "0.5",
            "--min_observations_per_landmark",
            "1",
            "--min_native_cosine",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output, map_location="cpu")
    assert artifact["gaussian_ids"].tolist() == [5, 9]
    assert torch.allclose(artifact["base_descriptors"], F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1))
    assert artifact["metadata"]["positive_pair_count"] == 1
    assert artifact["metadata"]["descriptor_mode"] == "solver_weighted_pair_cache_fusion"
    assert artifact["metadata"]["min_observations_per_landmark"] == 1
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "split_audit.json").exists()
    assert (tmp_path / "command.txt").exists()
    assert (tmp_path / "git_status.txt").exists()
    split_audit = json.loads((tmp_path / "split_audit.json").read_text())
    assert split_audit["test_split_used"] is False
    assert split_audit["official_test_used"] is False


def test_build_solver_weighted_feature_fusion_cli_defaults_to_v9_trust_region(tmp_path):
    pair_cache = tmp_path / "pairs.pt"
    output = tmp_path / "selected_landmark_descriptors.pt"
    torch.save(
        {
            "base_landmark_desc": F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1),
            "base_gaussian_id": torch.tensor([5, 9], dtype=torch.long),
            "query_desc": F.normalize(torch.tensor([[1.0, 0.1]], dtype=torch.float32), p=2, dim=-1),
            "landmark_id": torch.tensor([[0]], dtype=torch.long),
            "candidate_mask": torch.tensor([[True]]),
            "cosine": torch.tensor([[0.99]], dtype=torch.float32),
            "reprojection_error": torch.tensor([[0.5]], dtype=torch.float32),
            "metadata": {"source_split_name": "train", "feedback_bank_split_name": "selfmap_train"},
        },
        pair_cache,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_weighted_feature_fusion",
            "--pair_cache",
            str(pair_cache),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output, map_location="cpu")
    assert artifact["metadata"]["trust_alpha"] == 0.1
    assert artifact["metadata"]["min_native_cosine"] == 0.95
