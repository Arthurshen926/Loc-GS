import torch

from loc_gs.localization.ulfloc_scene_detector import (
    extract_ulfloc_scene_detector_keypoints,
    rerank_superpoint_keypoints_with_scene_detector,
)


class PeakDetector(torch.nn.Module):
    def __init__(self, heatmap: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("heatmap", heatmap.float())

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() == 3:
            return self.heatmap.unsqueeze(0)
        return self.heatmap.unsqueeze(0).unsqueeze(0).expand(feature_map.shape[0], -1, -1, -1)


def test_extract_ulfloc_scene_detector_keypoints_maps_coarse_cells_to_superpoint_pixels():
    feature_map = torch.zeros(256, 6, 8)
    heatmap = torch.zeros(6, 8)
    heatmap[2, 3] = 0.9
    heatmap[4, 1] = 0.7

    keypoints_xy, scores = extract_ulfloc_scene_detector_keypoints(
        feature_map,
        PeakDetector(heatmap),
        max_keypoints=2,
        nms_radius=0,
        descriptor_stride=8,
    )

    assert keypoints_xy.tolist() == [[27.5, 19.5], [11.5, 35.5]]
    assert torch.allclose(scores, torch.tensor([0.9, 0.7]))


def test_extract_ulfloc_scene_detector_keypoints_applies_threshold_and_topk():
    feature_map = torch.zeros(256, 4, 4)
    heatmap = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.05, 0.04, 0.03],
            [0.02, 0.01, 0.0, 0.0],
        ]
    )

    keypoints_xy, scores = extract_ulfloc_scene_detector_keypoints(
        feature_map,
        PeakDetector(heatmap),
        max_keypoints=1,
        nms_radius=0,
        score_threshold=0.75,
        descriptor_stride=8,
    )

    assert keypoints_xy.tolist() == [[3.5, 19.5]]
    assert torch.allclose(scores, torch.tensor([0.9]))


def test_rerank_superpoint_keypoints_preserves_coordinates_and_descriptors():
    feature_map = torch.zeros(256, 4, 4)
    heatmap = torch.zeros(4, 4)
    heatmap[2, 2] = 0.9
    heatmap[0, 0] = 0.2
    keypoints_xy = torch.tensor([[3.5, 3.5], [19.5, 19.5]], dtype=torch.float32)
    keypoint_scores = torch.tensor([0.95, 0.5], dtype=torch.float32)
    descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    out_keypoints, out_scores, out_descriptors = rerank_superpoint_keypoints_with_scene_detector(
        keypoints_xy,
        keypoint_scores,
        descriptors,
        feature_map,
        PeakDetector(heatmap),
        max_keypoints=1,
        descriptor_stride=8,
        blend_alpha=0.8,
    )

    assert out_keypoints.tolist() == [[19.5, 19.5]]
    assert out_descriptors.tolist() == [[0.0, 1.0]]
    assert out_scores.shape == (1,)


def test_rerank_superpoint_keypoints_can_preserve_native_safe_core():
    feature_map = torch.zeros(256, 4, 4)
    heatmap = torch.zeros(4, 4)
    heatmap[2, 2] = 0.9
    heatmap[0, 0] = 0.01
    keypoints_xy = torch.tensor([[3.5, 3.5], [19.5, 19.5]], dtype=torch.float32)
    keypoint_scores = torch.tensor([0.99, 0.5], dtype=torch.float32)
    descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    out_keypoints, _out_scores, out_descriptors = rerank_superpoint_keypoints_with_scene_detector(
        keypoints_xy,
        keypoint_scores,
        descriptors,
        feature_map,
        PeakDetector(heatmap),
        max_keypoints=2,
        descriptor_stride=8,
        blend_alpha=0.8,
        native_keep_fraction=0.5,
    )

    assert out_keypoints.tolist() == [[3.5, 3.5], [19.5, 19.5]]
    assert out_descriptors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
