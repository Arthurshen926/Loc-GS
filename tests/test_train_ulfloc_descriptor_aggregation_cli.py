from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _write_source_log(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    path.mkdir()
    features = F.normalize(torch.tensor([[0.7, 0.7], [0.0, 1.0]], dtype=torch.float32), p=2, dim=-1)
    sampled_idx = torch.tensor([10, 11], dtype=torch.long)
    with (path / "keypoints_features.pkl").open("wb") as handle:
        pickle.dump(features, handle)
    with (path / "keypoints_sampled_idx.pkl").open("wb") as handle:
        pickle.dump(sampled_idx, handle)
    (path / "config.yaml").write_text("sample: {}\n", encoding="utf-8")
    return features, sampled_idx


def _feedback_v4(*, split_name: str = "selfmap_train") -> dict:
    def row(label_role: str, descriptor: list[float], *, gaussian_id: int = 10, sampled_row: int = 0) -> dict:
        return {
            "scene": "ToyScene",
            "split_name": split_name,
            "query_id": f"q{gaussian_id}",
            "image_id": f"im{gaussian_id}",
            "gaussian_id": gaussian_id,
            "sampled_row": sampled_row,
            "query_keypoint_index": 0,
            "keypoint_xy": [5.0, 6.0],
            "query_xy_norm": [0.05, 0.12],
            "image_cell": 0,
            "query_descriptor": descriptor,
            "landmark_descriptor": [0.7, 0.7],
            "descriptor_score": 0.9,
            "descriptor_margin": 0.2,
            "detector_score": 0.7,
            "pnp_inlier": label_role in {"protected_support", "positive_inlier"},
            "reprojection_error_px": 1.0,
            "camera_xyz": [0.1, 0.2, 2.0],
            "depth_m": 2.0,
            "bearing": [0.0, 0.0, 1.0],
            "source_role": "baseline_trace",
            "label_role": label_role,
        }

    return {
        "schema_version": "sparse_solver_feedback_v4",
        "split_name": split_name,
        "split_audit": {"test_split_used": split_name == "test", "official_test_used": False},
        "correspondences": [
            row("protected_support", [1.0, 0.0]),
            row("positive_inlier", [0.9, 0.1]),
            row("harmful_negative", [0.0, 1.0]),
        ],
    }


def test_train_ulfloc_descriptor_aggregation_cli_writes_ulfloc_compatible_log(tmp_path: Path) -> None:
    source = tmp_path / "source_log"
    source_features, sampled_idx = _write_source_log(source)
    feedback = tmp_path / "feedback_v4.pt"
    torch.save(_feedback_v4(), feedback)
    output = tmp_path / "aggregated_log"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_descriptor_aggregation",
            "--source_log_dir",
            str(source),
            "--feedback_v4",
            str(feedback),
            "--output_log_dir",
            str(output),
            "--scene",
            "ShopFacade",
            "--steps",
            "80",
            "--lr",
            "0.2",
            "--min_native_cosine",
            "0.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with (output / "keypoints_features.pkl").open("rb") as handle:
        saved_features = torch.as_tensor(pickle.load(handle)).float()
    with (output / "keypoints_sampled_idx.pkl").open("rb") as handle:
        saved_idx = torch.as_tensor(pickle.load(handle)).long()
    metrics = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))

    assert torch.equal(saved_idx, sampled_idx)
    assert (output / "config.yaml").read_text(encoding="utf-8") == "sample: {}\n"
    assert not torch.allclose(saved_features[0], source_features[0])
    assert torch.allclose(saved_features[1], source_features[1])
    assert metrics["descriptor_mode"] == "ulfloc_solver_feedback_aggregation_v1"
    assert metrics["rebuilt_landmark_count"] == 1
    assert metrics["post_hoc_mean_shift_used"] is False
    assert manifest["schema_version"] == "ulfloc_descriptor_aggregation_manifest_v1"
    assert manifest["same_sampled_idx"] is True
    assert split_audit["test_split_used"] is False
    assert (output / "command.txt").exists()
    assert (output / "git_status.txt").exists()


def test_train_ulfloc_descriptor_aggregation_cli_rejects_test_feedback(tmp_path: Path) -> None:
    source = tmp_path / "source_log"
    _write_source_log(source)
    feedback = tmp_path / "feedback_v4.pt"
    torch.save(_feedback_v4(split_name="test"), feedback)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_descriptor_aggregation",
            "--source_log_dir",
            str(source),
            "--feedback_v4",
            str(feedback),
            "--output_log_dir",
            str(tmp_path / "aggregated_log"),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
