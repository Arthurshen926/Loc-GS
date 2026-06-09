import json
import subprocess
import sys

import torch


def test_build_sparse_feedback_detector_targets_from_allgauss_listwise_cache(tmp_path):
    pair_cache = tmp_path / "pairs.pt"
    output = tmp_path / "targets.pt"
    torch.save(
        {
            "query_yx": torch.tensor([[2.0, 3.0], [6.0, 7.0], [1.0, 1.0]], dtype=torch.float32),
            "image_id": ["img_a.png", "img_a.png", "img_b.png"],
            "label": torch.tensor([0, 2, 1], dtype=torch.long),
            "cosine": torch.tensor([[0.9, 0.1], [0.2, 0.1], [0.3, 0.8]], dtype=torch.float32),
            "candidate_mask": torch.tensor([[True, True], [False, False], [True, True]]),
            "reprojection_error": torch.tensor([[0.5, 8.0], [float("inf"), float("inf")], [5.0, 1.0]]),
            "metadata": {
                "format": "listwise",
                "topk": 2,
                "landmark_candidate_source": "all_gaussians",
                "source_split_name": "train",
                "split_audit": {"audit_status": "passed", "split_name": "train"},
            },
        },
        pair_cache,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_detector_targets",
            "--pair_cache",
            str(pair_cache),
            "--output",
            str(output),
            "--height",
            "10",
            "--width",
            "12",
            "--sigma_px",
            "0.75",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output, map_location="cpu")
    assert sorted(artifact["targets"]) == ["img_a.png", "img_b.png"]
    assert artifact["targets"]["img_a.png"]["positive_count"] == 1
    assert artifact["targets"]["img_a.png"]["heatmap"].shape == (10, 12)
    assert artifact["metadata"]["positive_rows"] == 2
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_cache"] == str(pair_cache)
    assert manifest["split_audit"]["audit_status"] == "passed"


def test_build_sparse_feedback_detector_targets_rejects_test_cache(tmp_path):
    pair_cache = tmp_path / "pairs_test.pt"
    output = tmp_path / "targets.pt"
    torch.save(
        {
            "query_yx": torch.tensor([[2.0, 3.0]], dtype=torch.float32),
            "image_id": ["img_test.png"],
            "label": torch.tensor([0], dtype=torch.long),
            "cosine": torch.tensor([[0.9]], dtype=torch.float32),
            "metadata": {"source_split_name": "test"},
        },
        pair_cache,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_detector_targets",
            "--pair_cache",
            str(pair_cache),
            "--output",
            str(output),
            "--height",
            "10",
            "--width",
            "12",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_build_sparse_feedback_detector_targets_uses_solver_validity(tmp_path):
    pair_cache = tmp_path / "pairs_solver.pt"
    output = tmp_path / "targets_solver.pt"
    torch.save(
        {
            "query_yx": torch.tensor([[2.0, 2.0], [6.0, 6.0]], dtype=torch.float32),
            "image_id": ["img.png", "img.png"],
            "label": torch.tensor([0, 0], dtype=torch.long),
            "cosine": torch.tensor([[0.9], [0.9]], dtype=torch.float32),
            "candidate_mask": torch.tensor([[True], [True]]),
            "reprojection_error": torch.tensor([[0.5], [0.5]], dtype=torch.float32),
            "solver_validity": torch.tensor([[1.0], [0.1]], dtype=torch.float32),
            "metadata": {
                "format": "listwise",
                "source_split_name": "train",
                "split_audit": {"audit_status": "passed", "split_name": "train"},
            },
        },
        pair_cache,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_sparse_feedback_detector_targets",
            "--pair_cache",
            str(pair_cache),
            "--output",
            str(output),
            "--height",
            "10",
            "--width",
            "10",
            "--score_sigma_px",
            "1000.0",
            "--solver_validity_power",
            "1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output, map_location="cpu")
    heatmap = artifact["targets"]["img.png"]["heatmap"]
    assert heatmap[2, 2] > heatmap[6, 6]
    assert artifact["metadata"]["solver_validity_power"] == 1.0
    assert artifact["targets"]["img.png"]["target_metadata"]["solver_validity_enabled"] is True
