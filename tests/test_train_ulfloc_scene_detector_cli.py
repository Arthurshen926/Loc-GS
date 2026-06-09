import json
import subprocess
import sys

import torch


def test_train_ulfloc_scene_detector_cli_rejects_test_targets(tmp_path):
    targets = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "split_name": "test",
            "targets": {},
            "metadata": {},
            "split_audit": {"split_name": "test", "test_split_used": True},
        },
        targets,
    )
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_scene_detector",
            "--detector_targets",
            str(targets),
            "--source_path",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--epochs",
            "0",
            "--seed",
            "123",
            "--device",
            "cpu",
            "--ulf_root",
            "/root/ULF-Loc",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_train_ulfloc_scene_detector_cli_zero_epoch_writes_audit_bundle(tmp_path):
    targets = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "schema_version": "ulfloc_detector_targets_from_solver_feedback_v1",
            "split_name": "train_self_map",
            "targets": {},
            "metadata": {
                "height": 20,
                "width": 30,
                "detector_target_storage": "points",
                "positive_detector_point_count": 0,
            },
            "split_audit": {"audit_status": "passed", "split_name": "train_self_map", "test_split_used": False},
        },
        targets,
    )
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_scene_detector",
            "--detector_targets",
            str(targets),
            "--source_path",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--epochs",
            "0",
            "--seed",
            "123",
            "--solver_validity_power",
            "1.0",
            "--device",
            "cpu",
            "--ulf_root",
            "/root/ULF-Loc",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "scene_detector.pth").exists()
    assert (out / "manifest.json").exists()
    assert (out / "metrics_summary.json").exists()
    assert (out / "split_audit.json").exists()
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["trained_image_count"] == 0
    assert metrics["epochs"] == 0
    assert metrics["seed"] == 123
    assert metrics["solver_validity_power"] == 1.0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 123
    assert manifest["solver_validity_power"] == 1.0
