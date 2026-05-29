import torch

from loc_gs.dense_support.dense_match_verifier import audit_dense_transition, verify_dense_matches


def test_dense_match_verifier_drops_ambiguous_high_descriptor_score_match():
    result = verify_dense_matches(
        query_yx=torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
        reference_yx=torch.tensor([[0.0, 0.0], [1.1, 1.0], [2.0, 2.0]], dtype=torch.float32),
        descriptor_scores=torch.tensor([0.95, 0.80, 0.50], dtype=torch.float32),
        local_geometry=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        lsf_support=torch.tensor([0.8, 0.2, 0.9], dtype=torch.float32),
        alpha_dominance=torch.tensor([0.9, 0.2, 0.8], dtype=torch.float32),
        ambiguity=torch.tensor([0.1, 0.9, 0.1], dtype=torch.float32),
        depth_uncertainty=torch.tensor([0.1, 0.8, 0.1], dtype=torch.float32),
        pose_leverage=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32),
        min_verifier_score=0.55,
        reweight_scores=True,
    )

    assert result["indices"].tolist() == [0, 2]
    assert result["metadata"]["dropped_count"] == 1
    assert result["metadata"]["valid_dense_match_ratio"] == 2 / 3
    assert torch.all(result["scores"] <= torch.tensor([0.95, 0.50]))


def test_dense_transition_audit_counts_sparse_correct_dense_wrong_and_worsened():
    summary = audit_dense_transition(
        [
            {"sparse_te_cm": 4.0, "sparse_re_deg": 1.0, "dense_te_cm": 12.0, "dense_re_deg": 1.0},
            {"sparse_te_cm": 20.0, "sparse_re_deg": 1.0, "dense_te_cm": 8.0, "dense_re_deg": 1.0},
            {"sparse_te_cm": 6.0, "sparse_re_deg": 1.0, "dense_te_cm": 30.0, "dense_re_deg": 1.0},
        ],
        sparse_correct_te_cm=5.0,
        dense_wrong_te_cm=10.0,
        worsen_margin_cm=5.0,
    )

    assert summary["sparse_correct_dense_wrong_count"] == 1
    assert summary["dense_worsened_count"] == 2
    assert summary["dense_improved_count"] == 1
