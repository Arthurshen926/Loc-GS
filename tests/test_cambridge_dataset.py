import json
from pathlib import Path

import numpy as np
from PIL import Image

from loc_gs.data.cambridge_dataset import CambridgeHybridDataset


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.full((8, 12, 3), 127, dtype=np.uint8))
    image.save(path)


def test_cambridge_dataset_loads_camera_json_train_items(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_rgb(scene_root / "processed" / "seq1" / "frame00001.png")
    cameras_json = tmp_path / "cameras.json"
    cameras_json.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "img_name": "seq1/frame00001.png",
                    "width": 12,
                    "height": 8,
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "fx": 10.0,
                    "fy": 11.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    ds = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=cameras_json,
        split="train",
        image_subdir="processed",
        image_height=4,
        image_width=6,
    )

    item = ds[0]
    assert item["rgb"].shape == (3, 4, 6)
    assert item["pose_w2c"].shape == (4, 4)
    assert np.allclose(item["pose_w2c"][:3, 3].numpy(), [-1.0, -2.0, -3.0])
    assert item["image_name"] == "seq1/frame00001.png"
    assert ds.scaled_intrinsics(6, 4)["fx"] == 5.0
    assert ds.scaled_intrinsics(6, 4)["cx"] == 2.75


def test_camera_json_missing_principal_point_uses_native_image_center(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_rgb(scene_root / "processed" / "seq1" / "frame00001.png")
    cameras_json = tmp_path / "cameras.json"
    cameras_json.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "img_name": "seq1/frame00001.png",
                    "width": 12,
                    "height": 8,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "fx": 10.0,
                    "fy": 10.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    ds = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=cameras_json,
        split="train",
        image_subdir="processed",
        image_height=4,
        image_width=6,
        cx=100.0,
        cy=100.0,
    )

    item = ds[0]
    assert np.allclose(item["K"].numpy()[:2, 2], [2.75, 1.75])


def test_cambridge_dataset_loads_dataset_test_split(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_rgb(scene_root / "processed" / "seq3" / "frame00008.png")
    (scene_root / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "\n"
        "seq3/frame00008.png 1.0 2.0 3.0 1.0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    ds = CambridgeHybridDataset(
        scene_root=scene_root,
        split="test",
        image_subdir="processed",
        image_height=4,
        image_width=6,
        fx=10.0,
        fy=11.0,
    )

    item = ds[0]
    assert item["rgb"].shape == (3, 4, 6)
    assert item["image_name"] == "seq3/frame00008.png"
    assert np.allclose(item["pose_w2c"][:3, 3].numpy(), [-1.0, -2.0, -3.0])


def test_cambridge_pose_file_quaternion_is_world_to_camera(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_rgb(scene_root / "processed" / "seq3" / "frame00008.png")
    (scene_root / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "\n"
        "seq3/frame00008.png 1.0 2.0 3.0 0.70710678 0.0 0.0 0.70710678\n",
        encoding="utf-8",
    )

    ds = CambridgeHybridDataset(
        scene_root=scene_root,
        split="test",
        image_subdir="processed",
        image_height=4,
        image_width=6,
        fx=10.0,
        fy=11.0,
    )

    item = ds[0]
    expected_R_w2c = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    expected_t = -expected_R_w2c @ np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(item["pose_w2c"][:3, :3].numpy(), expected_R_w2c, atol=1e-5)
    assert np.allclose(item["pose_w2c"][:3, 3].numpy(), expected_t, atol=1e-5)


def test_camera_json_filters_to_cambridge_split_files(tmp_path):
    scene_root = tmp_path / "ShopFacade"
    _write_rgb(scene_root / "processed" / "seq2" / "frame00001.png")
    _write_rgb(scene_root / "processed" / "seq3" / "frame00001.png")
    (scene_root / "dataset_train.txt").write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "\n"
        "seq2/frame00001.png 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    (scene_root / "dataset_test.txt").write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n"
        "\n"
        "seq3/frame00001.png 0 0 0 1 0 0 0\n",
        encoding="utf-8",
    )
    cameras_json = tmp_path / "cameras.json"
    cameras_json.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "img_name": "seq2/frame00001.png",
                    "width": 12,
                    "height": 8,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "fx": 10.0,
                    "fy": 10.0,
                },
                {
                    "id": 1,
                    "img_name": "seq3/frame00001.png",
                    "width": 12,
                    "height": 8,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "fx": 10.0,
                    "fy": 10.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    train = CambridgeHybridDataset(scene_root=scene_root, cameras_json=cameras_json, split="train")
    test = CambridgeHybridDataset(scene_root=scene_root, cameras_json=cameras_json, split="test")

    assert [record.image_name for record in train.records] == ["seq2/frame00001.png"]
    assert [record.image_name for record in test.records] == ["seq3/frame00001.png"]
