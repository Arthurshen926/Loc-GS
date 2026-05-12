import torch

from loc_gs.localization.lightglue_matcher import (
    lightglue_feature_name,
    make_lafs_from_yx,
    match_lightglue_descriptors,
)
from loc_gs.localization.matcher_registry import (
    DENSE_MATCHERS,
    DIM_PIPELINES,
    SPARSE_MATCHERS,
    normalize_dim_pipeline,
    resolve_sparse_dense_matchers,
)
from loc_gs.localization.rendered_keypoints import select_rendered_keypoints


def test_matcher_registry_exposes_dim_and_lightglue_choices():
    assert "lightglue" in SPARSE_MATCHERS
    assert "dim" in SPARSE_MATCHERS
    assert "lightglue_rendered" in DENSE_MATCHERS
    assert "loftr_rendered" in DENSE_MATCHERS
    assert "superpoint+lightglue" in DIM_PIPELINES
    assert "loftr" in DIM_PIPELINES
    assert normalize_dim_pipeline("SuperPoint+LightGlue") == "superpoint+lightglue"
    assert resolve_sparse_dense_matchers("stdloc_parity", "", "") == ("stdloc_parity", "stdloc_parity")
    assert resolve_sparse_dense_matchers("stdloc_parity", "lightglue", "lightglue_rendered") == (
        "lightglue",
        "lightglue_rendered",
    )


def test_lightglue_feature_name_maps_dim_pipelines():
    assert lightglue_feature_name("superpoint+lightglue") == "superpoint"
    assert lightglue_feature_name("aliked+lightglue") == "aliked"
    assert lightglue_feature_name("disk+lightglue") == "disk"


def test_make_lafs_from_yx_uses_xy_centers():
    yx = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    lafs = make_lafs_from_yx(yx, scale=2.0)

    assert lafs.shape == (1, 2, 2, 3)
    assert torch.allclose(lafs[0, :, 0, 2], torch.tensor([3.0, 5.0]))
    assert torch.allclose(lafs[0, :, 1, 2], torch.tensor([2.0, 4.0]))
    assert torch.allclose(lafs[0, :, 0, 0], torch.full((2,), 2.0))
    assert torch.allclose(lafs[0, :, 1, 1], torch.full((2,), 2.0))


def test_match_lightglue_descriptors_uses_injected_matcher_and_handles_empty():
    q_yx = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    r_yx = torch.tensor([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    q_desc = torch.eye(3)
    r_desc = torch.eye(3)

    class FakeMatcher:
        def __call__(self, desc1, desc2, lafs1, lafs2, hw1=None, hw2=None):
            assert desc1.shape == (3, 3)
            assert desc2.shape == (3, 3)
            assert lafs1.shape == (1, 3, 2, 3)
            assert hw1 == (8, 8)
            return torch.tensor([[0.9], [0.7]]), torch.tensor([[0, 0], [2, 1]])

    q_ids, r_ids, scores = match_lightglue_descriptors(
        q_yx,
        q_desc,
        r_yx,
        r_desc,
        image_hw=(8, 8),
        rendered_hw=(8, 8),
        matcher=FakeMatcher(),
    )

    assert q_ids.tolist() == [0, 2]
    assert r_ids.tolist() == [0, 1]
    assert torch.allclose(scores, torch.tensor([0.9, 0.7]))

    q_empty, r_empty, s_empty = match_lightglue_descriptors(
        q_yx[:1],
        q_desc[:1],
        r_yx[:1],
        r_desc[:1],
        matcher=FakeMatcher(),
    )
    assert q_empty.numel() == 0
    assert r_empty.numel() == 0
    assert s_empty.numel() == 0


def test_match_lightglue_descriptors_moves_injected_module_to_descriptor_device():
    q_yx = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    desc = torch.eye(2)

    class FakeModuleMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return self

        def forward(self, desc1, desc2, lafs1, lafs2, hw1=None, hw2=None):
            return torch.tensor([[1.0]]), torch.tensor([[0, 0]])

    matcher = FakeModuleMatcher()
    match_lightglue_descriptors(q_yx, desc, q_yx, desc, matcher=matcher)

    assert matcher.to_calls
    assert matcher.to_calls[0][0][0] == desc.device


def test_select_rendered_keypoints_supports_locability_detector_and_projected_sources():
    desc = torch.zeros(2, 4, 4)
    desc[:, 1, 2] = torch.tensor([1.0, 0.0])
    desc[:, 3, 0] = torch.tensor([0.0, 1.0])
    locability = torch.zeros(4, 4)
    locability[1, 2] = 0.9
    locability[3, 0] = 0.8

    loc = select_rendered_keypoints(
        desc,
        source="locability",
        locability=locability,
        max_keypoints=2,
        threshold=0.1,
        nms_radius=0,
    )
    assert loc.keypoints_yx.tolist() == [[1.0, 2.0], [3.0, 0.0]]
    assert loc.descriptors.shape == (2, 2)

    detector = torch.zeros(65, 1, 1)
    detector[4, 0, 0] = 10.0
    det = select_rendered_keypoints(
        desc,
        source="detector",
        detector_logits=detector,
        max_keypoints=1,
        threshold=0.0,
        nms_radius=0,
    )
    assert det.keypoints_yx.shape == (1, 2)
    assert det.descriptors.shape == (1, 2)

    projected = select_rendered_keypoints(
        desc,
        source="projected_gaussian",
        projected_yx=torch.tensor([[2.0, 1.0], [0.0, 3.0]]),
        projected_scores=torch.tensor([0.3, 0.7]),
        max_keypoints=1,
        threshold=0.0,
    )
    assert projected.keypoints_yx.tolist() == [[0.0, 3.0]]
