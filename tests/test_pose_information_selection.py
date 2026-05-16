import torch

from loc_gs.losses.pose_information_selection import (
    combined_pose_info_selector_loss,
    coverage_regularization_loss,
    hard_negative_suppression_loss,
    pnp_information_proxy_loss,
    selection_budget_loss,
)


def test_pnp_information_proxy_loss_prefers_high_inlier_high_info_points():
    inlier_prob = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    info_trace = torch.tensor([[10.0, 1.0]], dtype=torch.float32)
    good_logits = torch.tensor([[4.0, -4.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[-4.0, 4.0]], dtype=torch.float32)

    good = pnp_information_proxy_loss(good_logits, inlier_prob, info_trace)
    bad = pnp_information_proxy_loss(bad_logits, inlier_prob, info_trace)

    assert good < bad


def test_hard_negative_suppression_loss_increases_when_hard_negatives_are_selected():
    hard_negative_risk = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    selected_hard_negative = torch.tensor([[4.0, -4.0]], dtype=torch.float32)
    suppressed_hard_negative = torch.tensor([[-4.0, 4.0]], dtype=torch.float32)

    selected = hard_negative_suppression_loss(selected_hard_negative, hard_negative_risk)
    suppressed = hard_negative_suppression_loss(suppressed_hard_negative, hard_negative_risk)

    assert selected > suppressed


def test_selection_budget_loss_uses_target_fraction():
    half_selected = torch.tensor([[6.0, 6.0, -6.0, -6.0]], dtype=torch.float32)
    all_selected = torch.full((1, 4), 6.0, dtype=torch.float32)

    half_loss = selection_budget_loss(half_selected, target_fraction=0.5)
    all_loss = selection_budget_loss(all_selected, target_fraction=0.5)

    assert half_loss < all_loss


def test_coverage_regularization_penalizes_collapsed_selected_points():
    positions = torch.tensor(
        [[[0.0, 0.0], [0.05, 0.05], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    collapsed_logits = torch.tensor([[5.0, 5.0, -5.0, -5.0]], dtype=torch.float32)
    spread_logits = torch.tensor([[5.0, -5.0, 5.0, -5.0]], dtype=torch.float32)

    collapsed = coverage_regularization_loss(collapsed_logits, positions, sigma=0.25)
    spread = coverage_regularization_loss(spread_logits, positions, sigma=0.25)

    assert collapsed > spread


def test_combined_pose_info_selector_loss_handles_empty_mask_without_nan():
    logits = torch.randn(2, 3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    info_matrix = torch.eye(2).view(1, 1, 2, 2).expand(2, 3, 2, 2)
    out = combined_pose_info_selector_loss(
        logits,
        inlier_probability=torch.ones(2, 3),
        hard_negative_risk=torch.ones(2, 3),
        pose_information=info_matrix,
        positions_xy=torch.zeros(2, 3, 2),
        visibility_score=torch.ones(2, 3),
        mask=mask,
        budget_target_fraction=0.5,
    )

    assert set(out) >= {"loss", "pnp_information", "hard_negative", "budget", "coverage"}
    for value in out.values():
        assert torch.isfinite(value)
