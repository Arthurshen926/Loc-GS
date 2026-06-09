from types import SimpleNamespace

import pytest
import torch

from loc_gs.scripts.export_ulfloc_full_raw_projections import (
    gaussian_payload_from_ulfloc_model,
    intrinsic_from_fov,
    resolve_ulfloc_source_path_for_loader,
    view_record_from_ulfloc_camera,
)


def test_intrinsic_from_fov_uses_image_center_and_positive_focal():
    K = intrinsic_from_fov(fovx=1.0, fovy=0.8, width=100, height=80)

    assert K[0][0] > 0.0
    assert K[1][1] > 0.0
    assert K[0][2] == pytest.approx(50.0)
    assert K[1][2] == pytest.approx(40.0)
    assert K[2] == [0.0, 0.0, 1.0]


def test_view_record_from_ulfloc_camera_exports_world_to_camera_transpose():
    camera = SimpleNamespace(
        image_name="seq1/frame001.png",
        image_width=100,
        image_height=80,
        FoVx=1.0,
        FoVy=0.8,
        world_view_transform=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [2.0, 3.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )

    record = view_record_from_ulfloc_camera(camera, split_name="selfmap_train")

    assert record["source_view_id"] == "seq1/frame001.png"
    assert record["width"] == 100
    assert record["height"] == 80
    assert record["world_to_camera"][0][3] == pytest.approx(2.0)
    assert record["world_to_camera"][1][3] == pytest.approx(3.0)
    assert record["world_to_camera"][2][3] == pytest.approx(4.0)


def test_gaussian_payload_from_ulfloc_model_exports_full_raw_count_without_sampled_idx():
    gaussians = SimpleNamespace(
        get_xyz=torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=torch.float32),
        get_opacity=torch.tensor([[0.2], [0.8]], dtype=torch.float32),
    )

    payload = gaussian_payload_from_ulfloc_model(gaussians)

    assert payload["xyz"].shape == (2, 3)
    assert payload["opacity"].tolist() == pytest.approx([0.2, 0.8])
    assert payload["projection_source"] == "full_raw_gaussians"
    assert payload["source_gaussian_count"] == 2
    assert "sampled_idx" not in payload


def test_resolve_ulfloc_source_path_for_loader_adds_lowercase_cambridge_symlink(tmp_path):
    source = tmp_path / "Cambridge_stdloc" / "ShopFacade"
    source.mkdir(parents=True)

    resolved = resolve_ulfloc_source_path_for_loader(source, tmp_path / "links")

    assert "cambridge" in str(resolved)
    assert resolved.exists()
    assert resolved.resolve() == source.resolve()
