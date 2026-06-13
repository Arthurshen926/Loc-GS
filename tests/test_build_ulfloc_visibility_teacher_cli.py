import json
import subprocess
import sys

import torch


def test_build_ulfloc_visibility_teacher_cli_writes_synthetic_audit_bundle(tmp_path):
    projections = {
        "schema_version": "synthetic_ulfloc_sampled_projections_v1",
        "scene": "ShopFacade",
        "split_name": "train_selfmap",
        "height": 32,
        "width": 40,
        "projections": {
            "im1.png": [
                {"gaussian_id": 1, "keypoint_yx": [10.0, 20.0], "visible": True, "mask_valid": True},
                {"gaussian_id": 2, "keypoint_yx": [12.0, 22.0], "visible": True, "mask_valid": False},
            ],
            "im2.png": [
                {"gaussian_id": 3, "keypoint_yx": [4.0, 5.0], "visible": True, "mask_valid": True, "weight": 0.5},
            ],
        },
    }
    source = tmp_path / "projections.json"
    source.write_text(json.dumps(projections), encoding="utf-8")
    output_dir = tmp_path / "teacher"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_visibility_teacher",
            "--input_projections_json",
            str(source),
            "--output_dir",
            str(output_dir),
            "--split_name",
            "train_selfmap",
            "--scene",
            "ShopFacade",
            "--max_points_per_image",
            "1",
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
        "visibility_teacher.pt",
        "metrics_summary.json",
        "split_audit.json",
        "manifest.json",
        "command.txt",
        "git_status.txt",
    ):
        assert (output_dir / name).exists()

    artifact = torch.load(output_dir / "visibility_teacher.pt", map_location="cpu")
    assert artifact["split_name"] == "train_selfmap"
    assert artifact["targets"]["im1.png"]["gaussian_ids"].tolist() == [1]
    assert artifact["targets"]["im2.png"]["support_weights"].tolist() == [0.5]
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["input_projection_count"] == 3
    assert metrics["kept_projection_count"] == 2
    assert metrics["mask_invalid_projection_count"] == 1
    assert metrics["max_points_per_image"] == 1
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["test_split_used"] is False
    assert split_audit["projection_source"] == "input_projections_json"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ShopFacade"
    assert manifest["split"] == "train_selfmap"
    assert manifest["feedback_enabled"]["residual"] is False
    assert manifest["feedback_enabled"]["selector"] is False
    assert manifest["feedback_enabled"]["rho"] is False


def test_build_ulfloc_visibility_teacher_cli_rejects_test_split(tmp_path):
    source = tmp_path / "projections.json"
    source.write_text(
        json.dumps(
            {
                "split_name": "train_selfmap",
                "height": 8,
                "width": 8,
                "projections": {"im.png": []},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_visibility_teacher",
            "--input_projections_json",
            str(source),
            "--output_dir",
            str(tmp_path / "teacher"),
            "--split_name",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_build_ulfloc_visibility_teacher_parser_accepts_real_projection_args(tmp_path):
    from loc_gs.scripts.build_ulfloc_visibility_teacher import build_argparser

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--ulf_root",
            "/root/ULF-Loc",
            "--model_path",
            str(tmp_path / "model"),
            "--source_path",
            str(tmp_path / "source"),
            "--images",
            "processed",
            "--input_log_dir",
            str(tmp_path / "log"),
            "--cfg",
            str(tmp_path / "ulfloc.yaml"),
            "--output_dir",
            str(tmp_path / "teacher"),
            "--split_name",
            "train_selfmap",
            "--image_list",
            str(tmp_path / "selfmap_images.txt"),
            "--max_points_per_image",
            "4096",
        ]
    )

    assert args.scene == "ShopFacade"
    assert str(args.input_log_dir).endswith("log")
    assert str(args.cfg).endswith("ulfloc.yaml")
    assert str(args.image_list).endswith("selfmap_images.txt")
    assert args.max_points_per_image == 4096


def test_build_ulfloc_visibility_teacher_real_projection_target_is_vectorized():
    from argparse import Namespace
    import math

    from loc_gs.scripts.build_ulfloc_visibility_teacher import _project_sampled_gaussians_to_target

    camera = Namespace(
        image_name="im.png",
        image_width=100,
        image_height=100,
        FoVx=math.pi / 2.0,
        FoVy=math.pi / 2.0,
        world_view_transform=torch.eye(4),
    )
    gaussian_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    opacity = torch.tensor([0.1, 0.9, 0.8], dtype=torch.float32)

    entry, single_metrics, projection_metrics = _project_sampled_gaussians_to_target(
        gaussian_xyz=gaussian_xyz,
        opacity=opacity,
        sampled_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        camera=camera,
        masks=None,
        max_points_per_image=2,
    )

    assert entry["gaussian_ids"].tolist() == [1, 2]
    assert entry["positive_count"] == 2
    assert single_metrics["input_projection_count"] == 3
    assert single_metrics["kept_projection_count"] == 2
    assert single_metrics["dropped_by_image_cap_count"] == 1
    assert projection_metrics["projected_in_frame_count"] == 3


def test_build_ulfloc_visibility_teacher_real_projection_filters_render_invisible_sampled_points():
    from argparse import Namespace
    import math

    from loc_gs.scripts.build_ulfloc_visibility_teacher import _project_sampled_gaussians_to_target

    camera = Namespace(
        image_name="im.png",
        image_width=100,
        image_height=100,
        FoVx=math.pi / 2.0,
        FoVy=math.pi / 2.0,
        world_view_transform=torch.eye(4),
    )
    gaussian_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    render_visible_mask = torch.tensor([True, False, True], dtype=torch.bool)

    entry, single_metrics, projection_metrics = _project_sampled_gaussians_to_target(
        gaussian_xyz=gaussian_xyz,
        opacity=None,
        sampled_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        camera=camera,
        masks=None,
        max_points_per_image=0,
        render_visible_mask=render_visible_mask,
    )

    assert entry["gaussian_ids"].tolist() == [0, 2]
    assert entry["positive_count"] == 2
    assert single_metrics["invisible_projection_count"] == 1
    assert projection_metrics["render_visible_projection_count"] == 2
    assert projection_metrics["dropped_render_invisible_count"] == 1


def test_build_ulfloc_visibility_teacher_dataset_namespace_preserves_loader_symlink(tmp_path):
    from argparse import Namespace

    from loc_gs.scripts.build_ulfloc_visibility_teacher import _dataset_namespace

    real_source = tmp_path / "Cambridge_stdloc" / "ShopFacade"
    real_source.mkdir(parents=True)
    link = tmp_path / "links" / "cambridge_ShopFacade"
    link.parent.mkdir()
    link.symlink_to(real_source, target_is_directory=True)

    args = Namespace(
        sh_degree=3,
        source_path=link,
        feature_type=None,
        gaussian_type=None,
        model_path=tmp_path / "model",
        images="processed",
        resolution=-1,
        longest_edge=640,
        data_device="cpu",
    )
    dataset = _dataset_namespace(args, {"feature_type": "sp", "gaussian_type": "3dgs", "dense": {}})

    assert dataset.source_path == str(link)
    assert "cambridge" in dataset.source_path


def test_build_ulfloc_visibility_teacher_cli_real_projection_rejects_test_split(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_visibility_teacher",
            "--scene",
            "ShopFacade",
            "--ulf_root",
            "/root/ULF-Loc",
            "--model_path",
            str(tmp_path / "model"),
            "--source_path",
            str(tmp_path / "source"),
            "--images",
            "processed",
            "--input_log_dir",
            str(tmp_path / "log"),
            "--cfg",
            str(tmp_path / "ulfloc.yaml"),
            "--output_dir",
            str(tmp_path / "teacher"),
            "--split_name",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr
    assert "NotImplementedError" not in result.stderr


def test_build_ulfloc_visibility_teacher_cli_real_projection_missing_sampled_idx_is_clear(tmp_path):
    input_log_dir = tmp_path / "log"
    input_log_dir.mkdir()
    cfg = tmp_path / "ulfloc.yaml"
    cfg.write_text("sample: {}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_visibility_teacher",
            "--scene",
            "ShopFacade",
            "--ulf_root",
            "/root/ULF-Loc",
            "--model_path",
            str(tmp_path / "model"),
            "--source_path",
            str(tmp_path / "source"),
            "--images",
            "processed",
            "--input_log_dir",
            str(input_log_dir),
            "--cfg",
            str(cfg),
            "--output_dir",
            str(tmp_path / "teacher"),
            "--split_name",
            "train_selfmap",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "keypoints_sampled_idx.pkl" in result.stderr
    assert "NotImplementedError" not in result.stderr
