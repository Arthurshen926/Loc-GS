import json
import subprocess
import sys

import pytest
import torch

from loc_gs.training.ulfloc_solver_feedback_detector_targets import compose_detector_targets


def test_teacher_positive_and_negative_suppression_are_separate():
    teacher = {
        "q1": {
            "gaussian_ids": [1],
            "keypoint_yx": [[10.0, 20.0]],
            "support_weights": [1.0],
        }
    }
    impact = {
        "split_name": "train_selfmap",
        "detector_positive": {"q1": [{"gaussian_id": 3, "keypoint_xy": [50.0, 15.0], "weight": 2.0}]},
        "detector_negative": {"q1": [{"gaussian_id": 2, "keypoint_xy": [40.0, 30.0], "weight": 3.0}]},
    }

    out = compose_detector_targets(teacher, impact, height=64, width=80)

    assert out["q1"]["positive_count"] == 1
    assert out["q1"]["negative_count"] == 1
    assert out["q1"]["teacher_support_weights"][0] == 1.0
    assert out["q1"]["support_weights"][0] == 1.0
    assert out["q1"]["solver_positive_count"] == 1
    assert out["q1"]["solver_positive_gaussian_ids"].tolist() == [3]
    assert out["q1"]["solver_positive_keypoint_yx"].tolist() == [[15.0, 50.0]]
    assert out["q1"]["solver_positive_weights"][0] == 2.0
    assert out["q1"]["negative_keypoint_yx"].tolist() == [[30.0, 40.0]]
    assert out["q1"]["negative_weights"][0] == 3.0


def test_compose_detector_targets_keeps_superpoint_teacher_separate_from_visibility():
    visibility_teacher = {
        "split_name": "train_selfmap",
        "targets": {
            "q1": {
                "gaussian_ids": [1, 2],
                "keypoint_yx": [[10.0, 20.0], [12.0, 24.0]],
                "support_weights": [0.5, 0.75],
            }
        },
    }
    superpoint_teacher = {
        "split_name": "train_selfmap",
        "targets": {
            "q1": {
                "sp_teacher_keypoint_yx": [[30.0, 40.0]],
                "sp_teacher_weights": [0.9],
            }
        },
    }
    impact = {
        "split_name": "train_selfmap",
        "detector_positive": {},
        "detector_negative": {},
    }

    out = compose_detector_targets(
        visibility_teacher,
        impact,
        height=64,
        width=80,
        sp_teacher_targets=superpoint_teacher,
    )

    assert out["q1"]["keypoint_yx"].tolist() == [[10.0, 20.0], [12.0, 24.0]]
    assert out["q1"]["support_weights"].tolist() == [0.5, 0.75]
    assert out["q1"]["sp_teacher_keypoint_yx"].tolist() == [[30.0, 40.0]]
    assert out["q1"]["sp_teacher_weights"].tolist() == pytest.approx([0.9])
    assert out["q1"]["sp_teacher_count"] == 1
    assert out["q1"]["positive_count"] == 2
    assert out["q1"]["target_role"] == "superpoint_teacher_plus_visibility_plus_solver_residual"


def test_compose_detector_targets_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        compose_detector_targets({}, {"split_name": "test"}, height=8, width=8)


def test_compose_detector_targets_rejects_official_test_aliases():
    for split_name in ("official_test", "cambridge_test"):
        with pytest.raises(ValueError, match="test split"):
            compose_detector_targets({}, {"split_name": split_name}, height=8, width=8)


def test_build_ulfloc_solver_feedback_detector_targets_cli_writes_audit_bundle(tmp_path):
    teacher_path = tmp_path / "visibility_teacher.pt"
    impact_path = tmp_path / "impact_attribution.pt"
    output_dir = tmp_path / "detector"
    torch.save(
        {
            "schema_version": "ulfloc_visibility_teacher_v1",
            "scene": "ShopFacade",
            "split_name": "train_selfmap",
            "targets": {
                "q1.png": {
                    "gaussian_ids": torch.tensor([1], dtype=torch.long),
                    "keypoint_yx": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
                    "support_weights": torch.tensor([0.75], dtype=torch.float32),
                    "positive_count": 1,
                    "height": 64,
                    "width": 80,
                }
            },
            "metadata": {"height": 64, "width": 80},
            "split_audit": {"split_name": "train_selfmap", "test_split_used": False},
        },
        teacher_path,
    )
    torch.save(
        {
            "schema_version": "solver_feedback_impact_v1",
            "split_name": "train_selfmap",
            "detector_negative": {
                "q1.png": [
                    {"gaussian_id": 2, "keypoint_xy": [40.0, 30.0], "weight": 3.0},
                    {"gaussian_id": 3, "keypoint_xy": [400.0, 300.0], "weight": 5.0},
                ]
            },
            "detector_positive": {},
            "metadata": {"negative_landmark_count": 1},
        },
        impact_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_detector_targets",
            "--visibility_teacher",
            str(teacher_path),
            "--impact_attribution",
            str(impact_path),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
            "--checkpoint_path",
            str(tmp_path / "checkpoint.pth"),
            "--map_path",
            str(tmp_path / "map"),
            "--data_root",
            str(tmp_path / "data"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "detector_targets.pt",
        "metrics_summary.json",
        "split_audit.json",
        "manifest.json",
        "command.txt",
        "git_status.txt",
    ):
        assert (output_dir / name).exists()

    artifact = torch.load(output_dir / "detector_targets.pt", map_location="cpu")
    assert artifact["schema_version"] == "ulfloc_solver_feedback_detector_targets_v1"
    assert artifact["split_name"] == "train_selfmap"
    assert artifact["targets"]["q1.png"]["support_weights"].tolist() == [0.75]
    assert artifact["targets"]["q1.png"]["negative_gaussian_ids"].tolist() == [2]
    assert artifact["targets"]["q1.png"]["solver_positive_count"] == 0

    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["positive_detector_point_count"] == 1
    assert metrics["visibility_positive_point_count"] == 1
    assert metrics["sp_teacher_point_count"] == 0
    assert metrics["negative_suppression_point_count"] == 1
    assert metrics["out_of_bounds_negative_count"] == 1

    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["test_split_used"] is False
    assert split_audit["audit_status"] == "passed"

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ShopFacade"
    assert manifest["split_name"] == "train_selfmap"
    assert manifest["feedback_enabled"]["residual"] is False
    assert manifest["feedback_enabled"]["selector"] is False
    assert manifest["feedback_enabled"]["rho"] is False
    assert manifest["solver_feedback_negative_suppression_enabled"] is True


def test_build_ulfloc_solver_feedback_detector_targets_cli_rejects_test_split(tmp_path):
    teacher_path = tmp_path / "visibility_teacher.pt"
    impact_path = tmp_path / "impact_attribution.json"
    torch.save(
        {
            "schema_version": "ulfloc_visibility_teacher_v1",
            "split_name": "train_selfmap",
            "targets": {},
            "metadata": {"height": 8, "width": 8},
            "split_audit": {"split_name": "train_selfmap", "test_split_used": False},
        },
        teacher_path,
    )
    impact_path.write_text(json.dumps({"split_name": "test", "detector_negative": {}}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_detector_targets",
            "--visibility_teacher",
            str(teacher_path),
            "--impact_attribution",
            str(impact_path),
            "--output_dir",
            str(tmp_path / "detector"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_build_ulfloc_solver_feedback_detector_targets_cli_accepts_superpoint_teacher(tmp_path):
    teacher_path = tmp_path / "visibility_teacher.pt"
    sp_teacher_path = tmp_path / "superpoint_teacher.pt"
    impact_path = tmp_path / "impact_attribution.pt"
    output_dir = tmp_path / "detector"
    torch.save(
        {
            "schema_version": "ulfloc_visibility_teacher_v1",
            "scene": "ShopFacade",
            "split_name": "train_selfmap",
            "targets": {
                "q1.png": {
                    "gaussian_ids": torch.tensor([1], dtype=torch.long),
                    "keypoint_yx": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
                    "support_weights": torch.tensor([0.75], dtype=torch.float32),
                    "positive_count": 1,
                    "height": 64,
                    "width": 80,
                }
            },
            "metadata": {"height": 64, "width": 80},
            "split_audit": {"split_name": "train_selfmap", "test_split_used": False},
        },
        teacher_path,
    )
    torch.save(
        {
            "schema_version": "ulfloc_superpoint_teacher_v1",
            "scene": "ShopFacade",
            "split_name": "train_selfmap",
            "targets": {
                "q1.png": {
                    "sp_teacher_keypoint_yx": torch.tensor([[30.0, 40.0]], dtype=torch.float32),
                    "sp_teacher_weights": torch.tensor([0.9], dtype=torch.float32),
                    "height": 64,
                    "width": 80,
                }
            },
            "metadata": {"height": 64, "width": 80},
            "split_audit": {"split_name": "train_selfmap", "test_split_used": False},
        },
        sp_teacher_path,
    )
    torch.save(
        {
            "schema_version": "solver_feedback_impact_v1",
            "split_name": "train_selfmap",
            "detector_negative": {},
            "detector_positive": {},
        },
        impact_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_solver_feedback_detector_targets",
            "--visibility_teacher",
            str(teacher_path),
            "--superpoint_teacher",
            str(sp_teacher_path),
            "--impact_attribution",
            str(impact_path),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = torch.load(output_dir / "detector_targets.pt", map_location="cpu")
    entry = artifact["targets"]["q1.png"]
    assert entry["keypoint_yx"].tolist() == [[10.0, 20.0]]
    assert entry["sp_teacher_keypoint_yx"].tolist() == [[30.0, 40.0]]
    assert entry["sp_teacher_weights"].tolist() == pytest.approx([0.9])
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["visibility_positive_point_count"] == 1
    assert metrics["sp_teacher_point_count"] == 1
