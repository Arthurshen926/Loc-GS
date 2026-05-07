import torch

from utils.geometry_metrics import (
    chamfer_distance_stats,
    projected_depth_consistency,
)


def test_chamfer_distance_stats_reports_bidirectional_medians():
    source = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float32)

    stats = chamfer_distance_stats(source, target, chunk_size=1)

    assert stats["source_to_target_median"] == 1.0
    assert stats["target_to_source_median"] == 1.0
    assert stats["symmetric_chamfer_mean"] == 1.0


def test_projected_depth_consistency_uses_bilinear_depth_samples():
    depth = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32)
    points_xy = torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=torch.float32)
    point_depth = torch.tensor([4.0, 3.0], dtype=torch.float32)

    stats = projected_depth_consistency(depth, points_xy, point_depth)

    assert stats["count"] == 2
    assert stats["abs_error_median"] == 0.0
    assert stats["abs_error_mean"] == 0.0
