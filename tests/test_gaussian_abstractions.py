import struct
from pathlib import Path

import numpy as np

from loc_gs.core.camera import CameraIntrinsics
from loc_gs.gaussian.map import load_gaussian_map
from loc_gs.gaussian.renderer import GaussianRenderRequest
from loc_gs.gaussian.visibility import project_gaussian_visibility


def _write_gaussian_ply(path: Path) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(struct.pack("<3f", 0.0, 0.0, 4.0))
        handle.write(struct.pack("<3f", 10.0, 0.0, 4.0))
    return path


def test_gaussian_map_wraps_internal_landmark_loader(tmp_path: Path):
    gaussian_map = load_gaussian_map(_write_gaussian_ply(tmp_path / "point_cloud.ply"))

    assert gaussian_map.source_path.endswith("point_cloud.ply")
    assert gaussian_map.vertex_count == 2
    np.testing.assert_allclose(gaussian_map.lookup_xyz([0, 1]), [[0.0, 0.0, 4.0], [10.0, 0.0, 4.0]])


def test_gaussian_render_request_parses_manifest_record():
    record = {
        "scene": "GreatCourt",
        "split_name": "train_selfmap",
        "synthetic_query_id": "sim/GreatCourt/train_selfmap/000000",
        "render_engine": "3dgs",
        "render_contract": "posed_3dgs_camera_v1",
        "render_pose_c2w": [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "render_intrinsics": {"width": 640, "height": 480, "fx": 500.0, "fy": 501.0, "cx": 319.5, "cy": 239.5},
    }

    request = GaussianRenderRequest.from_manifest_record(record)

    assert request.synthetic_query_id == "sim/GreatCourt/train_selfmap/000000"
    assert request.render_engine == "3dgs"
    assert request.intrinsics.width == 640
    assert request.pose_c2w.shape == (4, 4)


def test_project_gaussian_visibility_reports_visible_indices():
    visibility = project_gaussian_visibility(
        np.array([[0.0, 0.0, 4.0], [10.0, 0.0, 4.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        pose_w2c=np.eye(4, dtype=np.float64),
        intrinsics=CameraIntrinsics(width=100, height=80, fx=50.0, fy=50.0, cx=49.5, cy=39.5),
    )

    assert visibility.visible_count == 1
    assert visibility.visible_indices.tolist() == [0]
    np.testing.assert_allclose(visibility.xy[0], [49.5, 39.5])
    assert visibility.depth.tolist() == [4.0, 4.0, -1.0]
