import torch

from loc_gs.localization.dim_image_matcher import match_loftr_images, rgb_to_grayscale


def test_rgb_to_grayscale_uses_expected_luma_weights():
    rgb = torch.zeros(1, 3, 2, 2)
    rgb[:, 0] = 1.0

    gray = rgb_to_grayscale(rgb)

    assert gray.shape == (1, 1, 2, 2)
    assert torch.allclose(gray, torch.full((1, 1, 2, 2), 0.299))


def test_match_loftr_images_uses_injected_matcher_filters_and_sorts_scores():
    query_rgb = torch.ones(1, 3, 20, 30)
    rendered_rgb = torch.ones(1, 3, 20, 30)

    class FakeMatcher:
        def __init__(self):
            self.seen_shapes = []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, batch):
            self.seen_shapes.append((tuple(batch["image0"].shape), tuple(batch["image1"].shape)))
            return {
                "keypoints0": torch.tensor(
                    [[2.0, 4.0], [5.0, 6.0], [10.0, 8.0]],
                    dtype=torch.float32,
                ),
                "keypoints1": torch.tensor(
                    [[3.0, 7.0], [6.0, 9.0], [9.0, 2.0]],
                    dtype=torch.float32,
                ),
                "confidence": torch.tensor([0.2, 0.9, 0.4], dtype=torch.float32),
            }

    matcher = FakeMatcher()
    query_yx, rendered_yx, scores = match_loftr_images(
        query_rgb,
        rendered_rgb,
        matcher=matcher,
        image_scale=0.5,
        min_confidence=0.3,
        max_matches=2,
    )

    assert matcher.seen_shapes == [((1, 1, 10, 15), (1, 1, 10, 15))]
    assert torch.allclose(query_yx, torch.tensor([[12.0, 10.0], [16.0, 20.0]]))
    assert torch.allclose(rendered_yx, torch.tensor([[18.0, 12.0], [4.0, 18.0]]))
    assert torch.allclose(scores, torch.tensor([0.9, 0.4]))


def test_match_loftr_images_returns_empty_when_no_valid_matches():
    class EmptyMatcher:
        def __call__(self, _batch):
            return {
                "keypoints0": torch.empty(0, 2),
                "keypoints1": torch.empty(0, 2),
                "confidence": torch.empty(0),
            }

    query_yx, rendered_yx, scores = match_loftr_images(
        torch.ones(1, 3, 8, 8),
        torch.ones(1, 3, 8, 8),
        matcher=EmptyMatcher(),
    )

    assert query_yx.shape == (0, 2)
    assert rendered_yx.shape == (0, 2)
    assert scores.numel() == 0
