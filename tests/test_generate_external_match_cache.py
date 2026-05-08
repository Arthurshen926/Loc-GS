import numpy as np
import torch

from loc_gs.scripts.generate_external_match_cache import geometric_inlier_mask


def test_geometric_inlier_mask_keeps_depth_consistent_reprojections():
    kpts_a_xy = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    kpts_b_xy = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    depth = torch.ones(4, 4)
    pose = torch.eye(4)
    K = torch.eye(3)

    mask = geometric_inlier_mask(
        kpts_a_xy,
        kpts_b_xy,
        depth,
        depth,
        pose,
        pose,
        K,
        reprojection_threshold_px=0.25,
        depth_tolerance=0.05,
    )

    assert mask.tolist() == [True, False]
