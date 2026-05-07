import ast
import sys
import types
import inspect
import textwrap

import torch


sys.modules.setdefault("poselib", types.SimpleNamespace())

import stdloc
from scene.gaussian_model import GaussianModel
from train_detector import normalize_match_scores


def test_lift_2d_to_3d_uses_bilinear_depth_for_subpixel_points():
    points2d = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    intrinsic = torch.eye(3, dtype=torch.float32)
    twc = torch.eye(4, dtype=torch.float32)
    depth_map = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32)

    points3d = stdloc.lift_2d_to_3d(
        points2d,
        intrinsic,
        twc,
        depth_map,
        interpolation="bilinear",
    )

    assert torch.allclose(points3d, torch.tensor([[4.0, 4.0, 4.0]]))


def test_soft_argmax_offsets_recovers_subpixel_local_peak():
    peak = torch.tensor([2.25, 1.5])
    x_weights = torch.tensor([0.0, 0.0, 0.75, 0.25])
    y_weights = torch.tensor([0.0, 0.5, 0.5, 0.0])
    weights = torch.outer(y_weights, x_weights).reshape(1, -1)
    scores = torch.log(weights.clamp_min(1e-9)) * 0.1

    offsets = stdloc.soft_argmax_offsets(scores, window_size=4, temperature=0.1)

    assert torch.allclose(offsets[0], peak, atol=1e-5)


def test_loc_sparse_uses_instance_config_not_main_global():
    tree = ast.parse(textwrap.dedent(inspect.getsource(stdloc.STDLoc.loc_sparse)))
    global_config_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    ]

    assert global_config_reads == []


def test_apply_landmark_prior_breaks_ambiguous_match_toward_stable_landmark():
    corr = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    prior = torch.tensor([0.1, 0.9], dtype=torch.float32)

    adjusted = stdloc.apply_landmark_prior(corr, prior, weight=0.1)

    assert adjusted[0, 1] > adjusted[0, 0]


def test_apply_rendered_prior_breaks_ambiguous_dense_match():
    corr = torch.zeros(1, 1, 3, dtype=torch.float32)
    rendered_prior = torch.tensor([[0.1, 0.5, 0.9]], dtype=torch.float32)

    adjusted = stdloc.apply_rendered_prior(corr, rendered_prior, weight=0.1)

    assert adjusted[0, 0, 2] > adjusted[0, 0, 0]


def test_sample_gaussians_copies_locability_logits():
    gaussians = GaussianModel(3)
    gaussians._xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    gaussians._loc_feature = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    gaussians._scaling = torch.zeros(4, 3)
    gaussians._opacity = torch.zeros(4, 1)
    gaussians._rotation = torch.zeros(4, 4)
    gaussians._features_dc = torch.zeros(4, 1, 3)
    gaussians._features_rest = torch.zeros(4, 15, 3)
    gaussians._locability_logit = torch.arange(4, dtype=torch.float32).reshape(4, 1)

    sampled = stdloc.sample_gaussians(gaussians, torch.tensor([3, 1]))

    assert torch.equal(sampled._locability_logit.squeeze(-1), torch.tensor([3.0, 1.0]))


def test_gaussian_model_exposes_sigmoid_locability():
    gaussians = GaussianModel(3)
    gaussians._locability_logit = torch.tensor([[-2.0], [2.0]])

    assert torch.allclose(gaussians.get_locability, torch.sigmoid(gaussians._locability_logit))


def test_normalize_match_scores_rewards_repeatable_high_score_landmarks():
    score_avg = torch.tensor([0.8, 0.8, 0.2], dtype=torch.float32)
    score_num = torch.tensor([20, 1, 20], dtype=torch.int32)

    locability = normalize_match_scores(score_avg, score_num)

    assert 0.0 <= locability.min() <= locability.max() <= 1.0
    assert locability[0] > locability[1]
    assert locability[0] > locability[2]


def test_select_eval_cameras_applies_stride_before_limit():
    cameras = list(range(10))

    selected = stdloc.select_eval_cameras(cameras, max_cameras=3, stride=2)

    assert selected == [0, 2, 4]


def test_select_eval_cameras_accepts_non_subscriptable_iterable():
    class CameraStream:
        def __iter__(self):
            return iter(range(10))

    selected = stdloc.select_eval_cameras(CameraStream(), max_cameras=3, stride=2)

    assert selected == [0, 2, 4]
