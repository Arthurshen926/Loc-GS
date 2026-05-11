import numpy as np
import torch
import torch.nn.functional as F

from loc_gs.localization import hybrid_localizer
from loc_gs.localization.hybrid_localizer import (
    flatten_rendered_landmarks,
    match_descriptors_topk,
    pose_error_cm_deg,
    refine_rendered_positions_softargmax,
    sample_descriptors_bilinear,
)


def test_sample_descriptors_bilinear_returns_normalized_vectors():
    desc = torch.zeros(2, 3, 3)
    desc[:, 1, 1] = torch.tensor([3.0, 4.0])
    keypoints = torch.tensor([[1.0, 1.0]])
    sampled = sample_descriptors_bilinear(desc, keypoints)
    assert sampled.shape == (1, 2)
    assert torch.allclose(sampled.norm(dim=-1), torch.ones(1))


def test_match_descriptors_topk_prefers_highest_cosine():
    query = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    landmark = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=-1)
    q_idx, lm_idx, scores = match_descriptors_topk(query, landmark, topk=1, threshold=-1.0)
    assert q_idx.tolist() == [0, 1]
    assert lm_idx.tolist() == [1, 0]
    assert torch.allclose(scores, torch.ones(2))


def test_match_descriptors_topk_can_filter_ambiguous_second_best():
    query = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    landmark = F.normalize(torch.tensor([[0.98, 0.02], [0.95, 0.05], [0.0, 1.0]]), dim=-1)
    q_idx, lm_idx, _scores = match_descriptors_topk(
        query,
        landmark,
        topk=1,
        threshold=-1.0,
        second_best_margin=0.02,
    )
    assert q_idx.tolist() == [1]
    assert lm_idx.tolist() == [2]


def test_refine_rendered_positions_softargmax_moves_to_local_descriptor_peak():
    desc = torch.zeros(2, 5, 5)
    desc[:, 2, 2] = torch.tensor([0.2, 0.8])
    desc[:, 2, 3] = torch.tensor([1.0, 0.0])
    desc = F.normalize(desc, dim=0)
    query = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    refined = refine_rendered_positions_softargmax(
        desc,
        query,
        torch.tensor([2 * 5 + 2]),
        window_radius=1,
        temperature=0.01,
    )

    assert refined.shape == (1, 2)
    assert torch.allclose(refined[0], torch.tensor([2.0, 3.0]), atol=1e-2)


def test_pose_error_cm_deg_zero_for_identical_pose():
    pose = np.eye(4, dtype=np.float32)
    te, ae = pose_error_cm_deg(pose, pose)
    assert te == 0.0
    assert ae == 0.0


def test_flatten_rendered_landmarks_filters_alpha_and_stride():
    desc = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    world = torch.randn(16, 3)
    alpha = torch.zeros(4, 4)
    alpha[::2, ::2] = 1.0
    xyz, flat_desc, ids = flatten_rendered_landmarks(desc, world, alpha, stride=2, alpha_threshold=0.5)
    assert xyz.shape == (4, 3)
    assert flat_desc.shape == (4, 2)
    assert ids.tolist() == [0, 2, 8, 10]


def test_poselib_solver_can_refine_pose_with_opencv_inliers(monkeypatch):
    points3d = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [0.5, 0.5, 6.0],
        ],
        dtype=np.float64,
    )
    keypoints_yx = np.stack([points3d[:, 1] / points3d[:, 2], points3d[:, 0] / points3d[:, 2]], axis=-1)
    K = np.eye(3, dtype=np.float64)
    init_pose = np.eye(4, dtype=np.float32)
    calls = {"solvepnp": 0}

    def fake_poselib(*_args, **_kwargs):
        return init_pose, 3

    def fake_solvepnp(obj, img, *_args, **_kwargs):
        calls["solvepnp"] += 1
        assert obj.shape[0] == points3d.shape[0]
        assert img.shape[0] == points3d.shape[0]
        return True, np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)

    monkeypatch.setattr(hybrid_localizer, "_solve_pnp_poselib", fake_poselib)
    monkeypatch.setattr(hybrid_localizer.cv2, "solvePnP", fake_solvepnp)

    pose, inliers = hybrid_localizer.solve_pnp_ransac(
        points3d,
        keypoints_yx,
        K,
        reprojection_error=2.0,
        solver="poselib",
        refine_poselib=True,
    )

    assert pose is not None
    assert inliers == points3d.shape[0]
    assert calls["solvepnp"] == 1


def test_poselib_solver_without_refine_does_not_require_opencv(monkeypatch):
    points3d = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
        ],
        dtype=np.float64,
    )
    keypoints_yx = np.stack([points3d[:, 1] / points3d[:, 2], points3d[:, 0] / points3d[:, 2]], axis=-1)
    K = np.eye(3, dtype=np.float64)
    init_pose = np.eye(4, dtype=np.float32)

    def fake_poselib(*_args, **_kwargs):
        return init_pose, 4

    def fail_solvepnp(*_args, **_kwargs):
        raise AssertionError("OpenCV fallback should not run when poselib succeeds")

    monkeypatch.setattr(hybrid_localizer, "_solve_pnp_poselib", fake_poselib)
    monkeypatch.setattr(hybrid_localizer.cv2, "solvePnPRansac", fail_solvepnp)

    pose, inliers = hybrid_localizer.solve_pnp_ransac(
        points3d,
        keypoints_yx,
        K,
        reprojection_error=2.0,
        solver="poselib",
        refine_poselib=False,
    )

    assert pose is init_pose
    assert inliers == 4
