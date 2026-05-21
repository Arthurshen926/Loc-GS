import torch

from loc_gs.diagnostics.pose_information import pose_information_summary


def test_pose_information_logdet_drops_for_collapsed_3d_support():
    xy = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    spread_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
        ],
        dtype=torch.float32,
    )
    collapsed_xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [0.0, 0.01, 1.0],
            [0.01, 0.01, 1.0],
        ],
        dtype=torch.float32,
    )

    spread = pose_information_summary(xy, spread_xyz)
    collapsed = pose_information_summary(xy, collapsed_xyz)

    assert spread["point_count"] == 4
    assert spread["logdet_H"] > collapsed["logdet_H"]
    assert spread["spatial_spread_3d"] > collapsed["spatial_spread_3d"]
    assert spread["min_eigenvalue_3d"] > collapsed["min_eigenvalue_3d"]

