import torch

from loc_gs.feedback.io import save_feedback_bank
from loc_gs.losses.hard_negative_descriptor import (
    feedback_weighted_descriptor_loss,
    hard_negative_margin_loss,
    residual_trust_region_loss,
)
from loc_gs.scripts.train_cambridge_hybrid import (
    build_argparser,
    feedback_bank_residual_loss,
    load_feedback_hard_negative_targets,
)


def test_hard_negative_margin_loss_prefers_positive_over_negative():
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pos = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    easy_neg = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    hard_neg = torch.tensor([[0.8, 0.2]], dtype=torch.float32)

    easy = hard_negative_margin_loss(query, pos, easy_neg, margin=0.2)
    hard = hard_negative_margin_loss(query, pos, hard_neg, margin=0.2)

    assert easy < hard


def test_residual_trust_region_loss_enforces_alpha_cap():
    base = torch.zeros(2, 4)
    inside = torch.full((2, 4), 0.01)
    outside = torch.full((2, 4), 0.5)

    inside_loss = residual_trust_region_loss(base, inside, alpha_max=0.1)
    outside_loss = residual_trust_region_loss(base, outside, alpha_max=0.1)

    assert inside_loss == 0
    assert outside_loss > inside_loss


def test_feedback_weighted_descriptor_loss_returns_components():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    pos = torch.tensor([[0.9, 0.1], [0.0, 1.0]], dtype=torch.float32)
    neg = torch.tensor([[0.8, 0.2], [1.0, 0.0]], dtype=torch.float32)
    base = torch.zeros(2, 2)
    residual = torch.tensor([[0.3, 0.0], [0.0, 0.01]], dtype=torch.float32)
    weights = torch.tensor([1.0, 0.0], dtype=torch.float32)

    out = feedback_weighted_descriptor_loss(
        query,
        pos,
        neg,
        base_desc=base,
        residual_desc=residual,
        hard_negative_weight=weights,
        margin=0.2,
        alpha_max=0.05,
        residual_trust_region_weight=0.5,
    )

    assert set(out) >= {"loss", "margin", "trust_region"}
    assert out["loss"] > 0
    assert out["trust_region"] > 0


def test_training_parser_keeps_hard_negative_residual_disabled_by_default():
    args = build_argparser().parse_args([])

    assert args.feedback_bank_path == ""
    assert args.enable_hard_negative_residual_loss is False
    assert args.hard_negative_margin == 0.2
    assert args.hard_negative_loss_weight == 0.0
    assert args.residual_trust_region_weight == 0.0


def test_feedback_bank_residual_loss_uses_synthetic_hard_negative_bank(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        [
            {
                "scene": "ShopFacade",
                "matched_gaussian_id": "1",
                "descriptor_score": 0.9,
                "pnp_inlier": False,
                "reprojection_error_px": 12.0,
            },
            {
                "scene": "ShopFacade",
                "matched_gaussian_id": "2",
                "descriptor_score": 0.9,
                "pnp_inlier": True,
                "pnp_success": True,
                "reprojection_error_px": 1.0,
            },
        ],
        {"scene": "ShopFacade", "split_name": "selfmap_train"},
    )

    targets = load_feedback_hard_negative_targets(bank_path)
    base_desc = torch.zeros(4, 3)
    residual_desc = base_desc.clone()
    residual_desc[1] = torch.tensor([0.2, 0.0, 0.0])

    out = feedback_bank_residual_loss(
        base_desc,
        residual_desc,
        targets,
        alpha_max=0.03,
        residual_trust_region_weight=1.0,
    )

    assert targets["gaussian_ids"].tolist() == [1]
    assert out["samples"].item() == 1
    assert out["loss"] > 0
    assert torch.isfinite(out["loss"])


def test_feedback_hard_negative_targets_reject_missing_split_name(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        [
            {
                "scene": "ShopFacade",
                "matched_gaussian_id": "1",
                "descriptor_score": 0.9,
                "pnp_inlier": False,
                "reprojection_error_px": 12.0,
            }
        ],
        {"scene": "ShopFacade", "split_name": ""},
    )

    try:
        load_feedback_hard_negative_targets(bank_path)
    except ValueError as exc:
        assert "split_name is required" in str(exc)
    else:
        raise AssertionError("expected missing split_name to be rejected")
