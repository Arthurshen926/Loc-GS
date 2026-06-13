import json
import subprocess
import sys

import pytest
import torch

from argparse import Namespace

from loc_gs.scripts.build_ulfloc_superpoint_teacher import _image_paths
from loc_gs.training.ulfloc_superpoint_teacher import build_superpoint_teacher_targets


def test_superpoint_teacher_image_list_reads_visual_landmark_first_token(tmp_path):
    root = tmp_path / "processed"
    (root / "seq2").mkdir(parents=True)
    image_path = root / "seq2" / "frame00001.png"
    image_path.write_bytes(b"fake")
    image_list = tmp_path / "dataset_test.txt"
    image_list.write_text(
        "Visual Landmark Dataset V1\n"
        "seq2/frame00001.png -1.0 -2.0 3.0 0.0 0.0 0.0 1.0\n",
        encoding="utf-8",
    )
    args = Namespace(image_list=image_list, image_glob="*.png", max_images=None)

    paths = _image_paths(args, root)

    assert paths == [image_path]


def test_build_superpoint_teacher_targets_converts_xy_to_yx_and_scores():
    detections = {
        "im1.png": [
            {"keypoint_xy": [20.0, 10.0], "score": 0.8},
            {"keypoint_yx": [12.0, 24.0], "keypoint_score": 0.6},
        ]
    }

    targets, metrics = build_superpoint_teacher_targets(
        detections,
        height=32,
        width=40,
        split_name="train_selfmap",
    )

    assert targets["im1.png"]["sp_teacher_keypoint_yx"].tolist() == [[10.0, 20.0], [12.0, 24.0]]
    assert targets["im1.png"]["sp_teacher_weights"].tolist() == pytest.approx([0.8, 0.6])
    assert targets["im1.png"]["sp_teacher_count"] == 2
    assert metrics["kept_keypoint_count"] == 2


def test_build_superpoint_teacher_targets_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        build_superpoint_teacher_targets({}, height=8, width=8, split_name="test")


def test_build_ulfloc_superpoint_teacher_cli_writes_synthetic_audit_bundle(tmp_path):
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_superpoint_detections_v1",
                "scene": "ShopFacade",
                "split_name": "train_selfmap",
                "height": 32,
                "width": 40,
                "detections": {
                    "im1.png": [
                        {"keypoint_xy": [20.0, 10.0], "score": 0.8},
                        {"keypoint_xy": [400.0, 10.0], "score": 0.9},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sp_teacher"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_superpoint_teacher",
            "--input_detections_json",
            str(detections_path),
            "--output_dir",
            str(output_dir),
            "--split_name",
            "train_selfmap",
            "--scene",
            "ShopFacade",
            "--data_root",
            str(tmp_path / "data"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "superpoint_teacher.pt",
        "metrics_summary.json",
        "split_audit.json",
        "manifest.json",
        "command.txt",
        "git_status.txt",
    ):
        assert (output_dir / name).exists()
    artifact = torch.load(output_dir / "superpoint_teacher.pt", map_location="cpu")
    assert artifact["schema_version"] == "ulfloc_superpoint_teacher_v1"
    assert artifact["targets"]["im1.png"]["sp_teacher_keypoint_yx"].tolist() == [[10.0, 20.0]]
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["input_keypoint_count"] == 2
    assert metrics["out_of_bounds_keypoint_count"] == 1
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["test_split_used"] is False
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ShopFacade"
    assert manifest["split_name"] == "train_selfmap"


def test_build_ulfloc_superpoint_teacher_cli_rejects_test_split(tmp_path):
    detections_path = tmp_path / "detections.json"
    detections_path.write_text(
        json.dumps({"split_name": "train_selfmap", "height": 8, "width": 8, "detections": {}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_superpoint_teacher",
            "--input_detections_json",
            str(detections_path),
            "--output_dir",
            str(tmp_path / "sp_teacher"),
            "--split_name",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
