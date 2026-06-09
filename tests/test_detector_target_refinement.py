import torch

from loc_gs.stdloc_native.detector_target_refinement import build_lsf_detector_target


def test_lsf_detector_target_refinement_boosts_supported_points_and_downweights_risky_points():
    heatmap, weight, metadata = build_lsf_detector_target(
        projected_yx=torch.tensor([[2.0, 2.0], [5.0, 5.0]], dtype=torch.float32),
        support_weights=torch.tensor([1.0, 1.0], dtype=torch.float32),
        hard_negative_risk=torch.tensor([0.0, 0.8], dtype=torch.float32),
        dense_worsen_risk=torch.tensor([0.0, 0.5], dtype=torch.float32),
        height=8,
        width=8,
        sigma_px=0.75,
    )

    assert heatmap.shape == (1, 1, 8, 8)
    assert weight.shape == (1, 1, 8, 8)
    assert heatmap[0, 0, 2, 2] > heatmap[0, 0, 5, 5]
    assert weight[0, 0, 2, 2] > 0.0
    assert metadata["point_count"] == 2
    assert metadata["risk_suppressed_count"] == 1


def test_lsf_detector_target_uses_solver_validity_weights():
    heatmap, weight, metadata = build_lsf_detector_target(
        projected_yx=torch.tensor([[2.0, 2.0], [5.0, 5.0]], dtype=torch.float32),
        support_weights=torch.tensor([1.0, 1.0], dtype=torch.float32),
        solver_validity_weights=torch.tensor([1.0, 0.1], dtype=torch.float32),
        solver_validity_power=1.0,
        height=8,
        width=8,
        sigma_px=0.75,
    )

    assert heatmap[0, 0, 2, 2] > heatmap[0, 0, 5, 5]
    assert weight[0, 0, 2, 2] > weight[0, 0, 5, 5]
    assert metadata["solver_validity_enabled"] is True
    assert metadata["low_solver_validity_count"] == 1
