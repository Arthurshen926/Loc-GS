import pytest
import torch

from loc_gs.stdloc_native.sparse_solver_set_selection import (
    SparseSetSelectionConfig,
    SparseSetSignals,
    per_query_support_from_solver_constraints,
    sparse_validation_signals_from_pnp_profile,
    select_sparse_solver_set_from_full_gaussians,
)


def test_full_gaussian_set_selection_is_variable_size_and_not_native_replacement():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [5.0, 0.0, 2.0],
                [0.0, 5.0, 8.0],
                [9.0, 9.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1]),
        visibility_score=torch.ones(5),
        mask_validity=torch.ones(5),
        solver_support=torch.tensor([0.1, 0.1, 1.0, 1.0, 0.0]),
        per_query_support={
            "q0": {2: 2.0, 3: 2.0},
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=5,
            min_marginal_gain=1.0,
            target_query_support=3.0,
            kc_weight=0.1,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=1.0,
            query_support_weight=2.0,
            geometry_weight=0.5,
        ),
    )

    assert result.sampled_idx.tolist() == [2, 3]
    assert result.metadata["budget_mode"] == "variable_size_marginal_gain"
    assert result.metadata["source"] == "full_gaussians"
    assert result.metadata["selected_count"] < 5
    assert "native_dropped_count" not in result.metadata
    assert "added_non_native_count" not in result.metadata
    assert result.metadata["selected_signal_stats"]["kc_score"]["mean"] == pytest.approx(0.65)
    assert result.metadata["selected_signal_stats"]["solver_support"]["mean"] == pytest.approx(1.0)


def test_full_gaussian_set_selection_rejects_unstable_masked_gaussians():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 1.0, 1.0]),
        visibility_score=torch.tensor([1.0, 1.0, 0.0]),
        mask_validity=torch.tensor([0.0, 1.0, 1.0]),
        solver_support=torch.tensor([10.0, 1.0, 1.0]),
        per_query_support={"q": {0: 10.0, 1: 1.0, 2: 1.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.1,
            min_mask_validity=0.5,
            min_visibility=0.5,
        ),
    )

    assert result.sampled_idx.tolist() == [1]
    assert result.metadata["invalid_mask_filtered_count"] == 1
    assert result.metadata["low_visibility_filtered_count"] == 1


def test_full_gaussian_set_selection_prefers_geometry_over_redundant_unary_score():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.05],
                [0.2, 0.0, 1.1],
                [4.0, 0.0, 4.0],
                [0.0, 4.0, 8.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 8.0, 2.0, 2.0]),
        visibility_score=torch.ones(5),
        mask_validity=torch.ones(5),
        solver_support=torch.ones(5),
        per_query_support={"q": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.2,
            max_per_spatial_cluster=1,
            spatial_cluster_size_m=1.0,
            kc_weight=0.05,
            solver_weight=0.1,
            query_support_weight=0.2,
            geometry_weight=3.0,
            depth_weight=1.0,
        ),
    )

    sampled = set(result.sampled_idx.tolist())
    assert 0 in sampled
    assert {3, 4}.issubset(sampled)
    assert 1 not in sampled
    assert 2 not in sampled
    assert result.metadata["cluster_cap_rejected_count"] >= 2


def test_full_gaussian_set_selection_prefers_solver_support_with_real_match_strength():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [4.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.ones(2),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.ones(2),
        per_query_support={"q": {0: 10.0, 1: 10.0}},
        support_match_strength=torch.tensor([0.1, 10.0]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            support_match_strength_weight=5.0,
        ),
    )

    assert result.sampled_idx.tolist() == [1]
    assert result.metadata["support_match_strength_weight"] == 5.0
    assert result.metadata["support_match_strength_nonzero_count"] == 2


def test_full_gaussian_set_selection_precision_fill_adds_stable_match_density_after_solver_saturation():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [4.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.2, 0.9, 0.8, 0.7, 0.1]),
        visibility_score=torch.ones(5),
        mask_validity=torch.ones(5),
        solver_support=torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]),
        support_match_strength=torch.tensor([0.1, 0.9, 0.8, 0.7, 0.0]),
        match_competition_risk=torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0]),
        per_query_support={"q": {0: 10.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=5,
            min_marginal_gain=0.0,
            target_query_support=4.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            main_min_query_gain=0.1,
            precision_fill_target_count=3,
            precision_fill_min_score=0.1,
            precision_fill_kc_weight=1.0,
            precision_fill_support_match_strength_weight=1.0,
            precision_fill_match_competition_risk_weight=2.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 1, 2]
    assert result.metadata["precision_fill_selected_count"] == 2
    assert result.metadata["precision_fill_target_count"] == 3


def test_full_gaussian_set_selection_penalizes_query_match_competition_risk():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [4.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.ones(2),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.ones(2),
        per_query_support={"q": {0: 10.0, 1: 10.0}},
        support_match_strength=torch.ones(2),
        match_competition_risk=torch.tensor([10.0, 0.0]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            support_match_strength_weight=1.0,
            match_competition_risk_weight=5.0,
        ),
    )

    assert result.sampled_idx.tolist() == [1]
    assert result.metadata["match_competition_risk_weight"] == 5.0
    assert result.metadata["match_competition_risk_nonzero_count"] == 1


def test_sparse_validation_hard_reject_exempts_protected_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [4.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.1, 10.0]),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.ones(2),
        per_query_support={"q_good": {0: 10.0}},
        validation_per_query_support={"q_good": {0: 10.0}},
        sparse_validation_risk=torch.tensor([5.0, 0.0]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            sparse_validation_risk_reject_threshold=0.1,
            sparse_validation_risk_protected_hard_reject_exempt=True,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=2.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            validation_query_prefill_target_fraction=1.0,
            validation_query_prefill_force_all=True,
        ),
    )

    assert result.sampled_idx.tolist() == [0]
    assert result.metadata["sparse_validation_protected_hard_reject_exempt"] is True
    assert result.metadata["sparse_validation_risk_hard_reject_exempt_count"] == 1
    assert result.metadata["sparse_validation_risk_hard_rejected_count"] == 0


def test_sparse_validation_hard_reject_exemption_is_separate_from_validation_prefill():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.ones(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={},
        validation_per_query_support={
            "validated_improvement": {0: 10.0},
            "protected_regression": {1: 10.0},
        },
        sparse_validation_risk=torch.tensor([5.0, 5.0, 0.0]),
        sparse_validation_hard_reject_exempt=torch.tensor([False, True, False]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            sparse_validation_risk_reject_threshold=0.1,
            sparse_validation_risk_protected_hard_reject_exempt=True,
            validation_query_prefill_target_fraction=1.0,
            validation_query_prefill_force_all=True,
        ),
    )

    selected = set(result.sampled_idx.tolist())
    assert 0 not in selected
    assert 1 in selected
    assert result.metadata["sparse_validation_risk_hard_reject_exempt_count"] == 1
    assert result.metadata["sparse_validation_risk_hard_rejected_count"] == 1


def test_sparse_validation_risk_can_exempt_reference_source_anchors():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 9.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        sparse_validation_risk=torch.tensor([8.0, 8.0, 0.0]),
        source_anchor_idx=torch.tensor([0]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            sparse_validation_risk_weight=1.0,
            sparse_validation_risk_reject_threshold=5.0,
            sparse_validation_risk_source_anchor_exempt=True,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    selected = set(result.sampled_idx.tolist())
    assert 0 in selected
    assert 1 not in selected
    assert result.metadata["sparse_validation_risk_source_anchor_exempt"] is True
    assert result.metadata["sparse_validation_risk_source_anchor_exempt_count"] == 1
    assert result.metadata["sparse_validation_risk_hard_rejected_count"] == 1
    assert result.metadata["sparse_validation_risk_nonzero_count"] == 1
    assert result.metadata["sparse_validation_risk_original_nonzero_count"] == 2


def test_sparse_validation_source_anchor_risk_exempt_requires_source_anchor_idx():
    signals = SparseSetSignals(
        xyz=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        kc_score=torch.ones(1),
        visibility_score=torch.ones(1),
        mask_validity=torch.ones(1),
        solver_support=torch.ones(1),
        sparse_validation_risk=torch.tensor([8.0]),
    )

    with pytest.raises(ValueError, match="source_anchor_idx"):
        select_sparse_solver_set_from_full_gaussians(
            signals,
            SparseSetSelectionConfig(
                sparse_validation_risk_weight=1.0,
                sparse_validation_risk_source_anchor_exempt=True,
            ),
        )


def test_sparse_validation_profile_exports_selector_signals():
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v2",
        "split_name": "train_dev",
        "attribution_status": "attributed",
        "landmark_regression_risk": {"1": 7.5, "9": 99.0},
        "protected_per_query_support": {"q_easy": {"1": 4.0, "2": 3.0}},
        "validated_per_query_support": {"q_hard": {"3": 5.0}},
        "protected_per_query_observations": {
            "q_easy": {
                "1": {
                    "xy_norm": [0.2, 0.3],
                    "depth_m": 4.0,
                    "reprojection_error_px": 0.5,
                }
            }
        },
        "validated_per_query_observations": {
            "q_hard": {
                "3": {
                    "xy_norm": [0.6, 0.7],
                    "depth_m": 8.0,
                    "descriptor_margin": 0.4,
                }
            }
        },
    }

    signals = sparse_validation_signals_from_pnp_profile(profile, num_gaussians=5)

    assert signals["sparse_validation_risk"].tolist() == [0.0, 7.5, 0.0, 0.0, 0.0]
    assert signals["sparse_validation_hard_reject_exempt"].tolist() == [False, True, True, True, False]
    assert signals["validation_per_query_support"] == {
        "protected:q_easy": {1: 4.0, 2: 3.0},
        "validated:q_hard": {3: 5.0},
    }
    assert signals["per_query_observations"]["protected:q_easy"][1]["depth_m"] == 4.0
    assert signals["per_query_observations"]["validated:q_hard"][3]["descriptor_margin"] == 0.4
    assert signals["metadata"]["protected_support_landmark_count"] == 2
    assert signals["metadata"]["validated_support_landmark_count"] == 1
    assert signals["metadata"]["attribution_status"] == "attributed"


def test_sparse_validation_profile_rejects_metric_only_when_required_for_main_selector():
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v2",
        "split_name": "train_dev",
        "attribution_status": "metric_only",
        "landmark_regression_risk": {"1": 7.5},
    }

    with pytest.raises(ValueError, match="attribution_status='attributed'"):
        sparse_validation_signals_from_pnp_profile(
            profile,
            num_gaussians=5,
            require_attribution=True,
        )


def test_sparse_validation_profile_support_scope_keeps_only_protected_regressions():
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v1",
        "split_name": "train_dev",
        "protected_regression_query_ids": ["q_regress"],
        "protected_per_query_support": {
            "q_easy": {"1": 4.0},
            "q_regress": {"2": 6.0},
        },
        "validated_per_query_support": {"q_hard": {"3": 5.0}},
    }

    signals = sparse_validation_signals_from_pnp_profile(
        profile,
        num_gaussians=5,
        support_scope="protected_regressions",
    )

    assert signals["validation_per_query_support"] == {
        "protected:q_regress": {2: 6.0},
        "validated:q_hard": {3: 5.0},
    }
    assert signals["sparse_validation_hard_reject_exempt"].tolist() == [False, False, True, True, False]


def test_sparse_validation_profile_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        sparse_validation_signals_from_pnp_profile(
            {
                "split_name": "test",
                "landmark_regression_risk": {"1": 1.0},
            },
            num_gaussians=4,
        )


def test_validation_query_prefill_force_all_keeps_all_protected_inliers():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([3.0, 2.0, 1.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"q_good": {0: 10.0, 1: 10.0, 2: 10.0}},
        validation_per_query_support={"q_good": {0: 10.0, 1: 10.0, 2: 10.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=100.0,
            target_query_support=10.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            validation_query_prefill_target_fraction=0.5,
            validation_query_prefill_force_all=True,
        ),
    )

    assert set(result.sampled_idx.tolist()) == {0, 1, 2}
    assert result.metadata["validation_query_prefill_force_all"] is True
    assert result.metadata["validation_query_prefill_selected_count"] == 3


def test_match_strength_candidate_pool_keeps_real_support_outside_static_topk():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"q": {2: 10.0}},
        support_match_strength=torch.tensor([0.0, 0.0, 10.0]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            candidate_pool_size=1,
            candidate_pool_match_strength_top_k=1,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            support_match_strength_weight=0.1,
        ),
    )

    assert result.metadata["candidate_pool_static_count"] == 1
    assert result.metadata["candidate_pool_match_strength_added_count"] == 1
    assert result.metadata["candidate_count"] == 2


def test_query_match_strength_candidate_pool_keeps_query_specific_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"q": {2: 10.0}},
        per_query_match_strength={"q": {2: 10.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            candidate_pool_size=1,
            candidate_pool_query_match_strength_top_k=1,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_match_strength_weight=1.0,
        ),
    )

    assert result.metadata["candidate_pool_static_count"] == 1
    assert result.metadata["candidate_pool_query_match_strength_added_count"] == 1
    assert result.metadata["candidate_count"] == 2


def test_query_match_strength_can_pass_main_query_gain_gate_as_context_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 0.1]),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.zeros(2),
        per_query_support={},
        per_query_match_strength={"protected": {1: 10.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_landmarks=0,
            min_marginal_gain=0.0,
            main_min_query_gain=1.0,
            candidate_pool_size=1,
            candidate_pool_query_match_strength_top_k=1,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_match_strength_weight=1.0,
        ),
    )

    assert result.sampled_idx.tolist() == [1]
    assert result.metadata["candidate_pool_query_match_strength_added_count"] == 1
    assert result.metadata["main_query_match_context_gate_accepted_count"] == 1


def test_query_match_dominance_rejects_same_query_stealer_with_high_unary_score():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [4.0, 0.0, 2.0],
                [0.0, 4.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 0.5]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 4.0}},
        per_query_match_strength={"hard": {0: 10.0}},
        per_query_match_competition_risk={"hard": {1: 20.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.1,
            main_dynamic_lookahead=3,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_match_strength_weight=4.0,
            query_match_competition_weight=2.0,
            query_match_competition_ratio_weight=4.0,
        ),
    )

    sampled = set(result.sampled_idx.tolist())
    assert 0 in sampled
    assert 1 not in sampled
    assert result.metadata["query_match_competition_risk_query_count"] == 1
    assert result.metadata["final_query_match_strength"]["hard"] > 0.0
    assert result.metadata["final_query_match_risk"]["hard"] == 0.0


def test_query_match_reference_risk_penalty_preserves_good_sparse_basin():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 2.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"easy": {0: 8.0}},
        per_query_match_strength={"easy": {0: 8.0}},
        per_query_match_competition_risk={"easy": {1: 6.0}},
        per_query_match_competition_risk_reference={"easy": 1.0},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.1,
            target_query_support=8.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=2.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_match_strength_weight=2.0,
            query_match_risk_reference_margin=0.5,
            query_match_risk_reference_weight=10.0,
        ),
    )

    sampled = set(result.sampled_idx.tolist())
    assert 0 in sampled
    assert 1 not in sampled
    assert 2 in sampled
    assert result.metadata["query_match_risk_reference_query_count"] == 1


def test_query_match_nonreference_competitor_penalty_blocks_new_sparse_basin_stealer():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 2.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"easy": {0: 8.0}},
        per_query_match_strength={"easy": {0: 8.0}},
        per_query_match_competition_risk={"easy": {1: 1.0, 2: 1.0}},
        per_query_match_competition_reference_landmarks={"easy": {2}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.1,
            target_query_support=8.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=2.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_match_strength_weight=2.0,
            query_match_nonreference_competition_weight=20.0,
        ),
    )

    sampled = set(result.sampled_idx.tolist())
    assert 0 in sampled
    assert 1 not in sampled
    assert 2 in sampled
    assert result.metadata["query_match_competition_reference_query_count"] == 1


def test_final_prune_query_match_reference_removes_non_support_competitors():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 9.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"easy": {0: 8.0, 2: 4.0}},
        per_query_match_strength={"easy": {0: 8.0, 2: 4.0}},
        per_query_match_competition_risk={"easy": {1: 5.0, 2: 2.0}},
        per_query_match_competition_risk_reference={"easy": 2.0},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            target_query_support=8.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            final_prune_query_match_risk_reference=True,
            final_prune_min_keep_count=2,
        ),
    )

    assert set(result.sampled_idx.tolist()) == {0, 2}
    assert result.metadata["final_pruned_query_match_risk_reference_count"] == 1
    assert result.metadata["final_query_match_risk"]["easy"] == 2.0
    assert result.metadata["final_query_match_risk_over_reference_count"] == 0


def test_final_prune_nonreference_query_match_competition_removes_swapped_competitor():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 9.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"easy": {0: 8.0}},
        per_query_match_strength={"easy": {0: 8.0}},
        per_query_match_competition_risk={"easy": {1: 1.0, 2: 1.0}},
        per_query_match_competition_reference_landmarks={"easy": {2}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            target_query_support=8.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            final_prune_nonreference_query_match_competition=True,
            final_prune_min_keep_count=2,
        ),
    )

    assert set(result.sampled_idx.tolist()) == {0, 2}
    assert result.metadata["final_pruned_nonreference_query_match_competition_count"] == 1


def test_solver_constraints_are_converted_to_positive_set_evidence():
    constraints = {
        "hard_query_ids": ["hard"],
        "candidate_gain": {
            "10": {"hard": {"support": 2.0, "viable_tuple_mass": 3.0}},
        },
        "source_loss": {
            "20": {"hard": {"support": 5.0, "viable_tuple_mass": 7.0}},
        },
    }

    support = per_query_support_from_solver_constraints(constraints)

    assert support == {
        "hard": {
            10: 5.0,
            20: 12.0,
        }
    }


def test_query_prefill_keeps_low_static_solver_backed_query_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [5.0, 0.0, 3.0],
                [5.2, 0.0, 4.0],
                [7.0, 0.0, 6.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 8.0, 0.1, 0.1, 0.1]),
        visibility_score=torch.ones(6),
        mask_validity=torch.ones(6),
        solver_support=torch.tensor([0.1, 0.1, 0.1, 1.0, 1.0, 0.5]),
        per_query_support={
            "easy": {0: 4.0, 1: 4.0, 2: 4.0},
            "hard": {3: 2.0, 4: 2.0, 5: 1.0},
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=4,
            min_marginal_gain=0.0,
            target_query_support=4.0,
            query_prefill_target_fraction=1.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    sampled = set(result.sampled_idx.tolist())
    assert {3, 4}.issubset(sampled)
    assert result.metadata["query_prefill_selected_count"] >= 2
    assert result.metadata["final_query_coverage"]["hard"] >= 4.0


def test_validation_support_prefill_keeps_selfmap_validated_support_before_main_selection():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [5.0, 0.0, 3.0],
                [7.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.1, 0.1]),
        visibility_score=torch.ones(4),
        mask_validity=torch.ones(4),
        solver_support=torch.zeros(4),
        validation_per_query_support={"q_hard": {2: 2.0, 3: 2.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=2.0,
            validation_query_prefill_target_fraction=1.0,
            validation_query_prefill_max_candidates_per_query=4,
            validation_query_prefill_min_cells_per_query=2,
            validation_query_prefill_min_depth_bins_per_query=2,
            augmentation_min_kc_score=0.5,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
        ),
    )

    assert set(result.sampled_idx.tolist()) == {2, 3}
    assert result.metadata["validation_query_prefill_selected_count"] == 2
    assert result.metadata["validation_query_evidence_query_count"] == 1


def test_query_prefill_does_not_count_existing_selected_points_as_cluster_rejections():
    signals = SparseSetSignals(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]], dtype=torch.float32),
        kc_score=torch.tensor([1.0, 0.9]),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.ones(2),
        per_query_support={"q": {0: 1.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=1.0,
            query_prefill_target_fraction=1.0,
            max_per_spatial_cluster=0,
        ),
    )

    assert result.metadata["query_prefill_selected_count"] == 1
    assert result.metadata["cluster_cap_rejected_count"] == 0


def test_soft_source_anchor_prior_is_scaled_to_query_utility():
    signals = SparseSetSignals(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
        kc_score=torch.ones(2),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.ones(2),
        per_query_support={"q": {0: 99.0, 1: 100.0}},
        source_anchor_idx=torch.tensor([0], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            target_query_support=100.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=2.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            source_anchor_mode="soft",
            source_anchor_weight=1.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0]
    assert result.metadata["source_anchor_soft_prior_scale"] == 100.0


def test_soft_source_anchor_prior_satisfies_main_query_gain_gate():
    signals = SparseSetSignals(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
        kc_score=torch.ones(2),
        visibility_score=torch.ones(2),
        mask_validity=torch.ones(2),
        solver_support=torch.zeros(2),
        per_query_support={},
        source_anchor_idx=torch.tensor([0], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            main_min_query_gain=0.05,
            target_query_support=8.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            source_anchor_mode="soft",
            source_anchor_weight=1.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0]


def test_kc_anchor_prefill_preserves_high_keypoint_consensus_landmarks():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.1, 0.1]),
        visibility_score=torch.ones(4),
        mask_validity=torch.ones(4),
        solver_support=torch.tensor([0.0, 0.0, 10.0, 9.0]),
        per_query_support={"q": {2: 2.0, 3: 2.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            kc_anchor_count=2,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=1.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    sampled = result.sampled_idx.tolist()
    assert sampled[:2] == [0, 1]
    assert result.metadata["kc_anchor_selected_count"] == 2


def test_kc_anchor_prefill_can_enforce_spatial_coverage_without_global_cluster_cap():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [5.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 8.0, 1.0]),
        visibility_score=torch.ones(4),
        mask_validity=torch.ones(4),
        solver_support=torch.ones(4),
        per_query_support={},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            spatial_cluster_size_m=1.0,
            max_per_spatial_cluster=0,
            kc_anchor_count=2,
            kc_anchor_max_per_spatial_cluster=1,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 3]
    assert result.metadata["kc_anchor_selected_count"] == 2
    assert result.metadata["kc_anchor_max_per_spatial_cluster"] == 1


def test_query_prefill_can_enforce_per_query_spatial_coverage_without_global_cluster_cap():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [5.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(4),
        visibility_score=torch.ones(4),
        mask_validity=torch.ones(4),
        solver_support=torch.ones(4),
        per_query_support={"q": {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=7.0,
            spatial_cluster_size_m=1.0,
            max_per_spatial_cluster=0,
            query_prefill_target_fraction=1.0,
            query_prefill_max_per_spatial_cluster=1,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 3]
    assert result.metadata["query_prefill_selected_count"] == 2
    assert result.metadata["query_prefill_max_per_spatial_cluster"] == 1


def test_query_prefill_rewards_per_query_geometry_diversity():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.1],
                [5.0, 0.0, 7.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 2.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=20.0,
            spatial_cluster_size_m=1.0,
            depth_bin_size_m=2.0,
            query_prefill_target_fraction=1.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_geometry_weight=20.0,
            query_depth_weight=10.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["query_geometry_weight"] == 20.0
    assert result.metadata["query_depth_weight"] == 10.0


def test_query_prefill_uses_query_observation_cells_when_available():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.2, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 2.0}},
        per_query_observations={
            "hard": {
                0: {"cell": [0, 0], "depth_bin": 0},
                1: {"cell": [0, 0], "depth_bin": 0},
                2: {"cell": [4, 0], "depth_bin": 0},
            }
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=20.0,
            spatial_cluster_size_m=10.0,
            depth_bin_size_m=10.0,
            query_prefill_target_fraction=1.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_geometry_weight=20.0,
            query_depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["query_observation_entry_count"] == 3


def test_query_prefill_min_cell_and_depth_targets_continue_after_support_saturation():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.2, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 100.0, 1: 90.0, 2: 1.0}},
        per_query_observations={
            "hard": {
                0: {"cell": [0, 0], "depth_bin": 0},
                1: {"cell": [0, 0], "depth_bin": 0},
                2: {"cell": [4, 0], "depth_bin": 3},
            }
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=4.0,
            spatial_cluster_size_m=10.0,
            depth_bin_size_m=10.0,
            query_prefill_target_fraction=1.0,
            query_prefill_min_cells_per_query=2,
            query_prefill_min_depth_bins_per_query=2,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_geometry_weight=0.0,
            query_depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["query_prefill_min_cells_per_query"] == 2
    assert result.metadata["query_prefill_min_depth_bins_per_query"] == 2
    assert result.metadata["final_query_cell_counts"]["hard"] == 2
    assert result.metadata["final_query_depth_bin_counts"]["hard"] == 2


def test_main_greedy_can_require_query_solver_gain_after_anchors():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 10.0, 9.0, 0.1]),
        visibility_score=torch.ones(4),
        mask_validity=torch.ones(4),
        solver_support=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        per_query_support={"hard": {3: 2.0}},
        source_anchor_idx=torch.tensor([0], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=4,
            min_marginal_gain=0.0,
            main_min_query_gain=0.5,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 3]
    assert result.metadata["main_min_query_gain"] == 0.5
    assert result.metadata["main_query_gain_gate_rejected_count"] == 2


def test_dynamic_lookahead_uses_bounded_pending_window():
    count = 50
    signals = SparseSetSignals(
        xyz=torch.stack(
            [
                torch.arange(count, dtype=torch.float32),
                torch.zeros(count, dtype=torch.float32),
                torch.ones(count, dtype=torch.float32),
            ],
            dim=1,
        ),
        kc_score=torch.linspace(float(count), 1.0, count),
        visibility_score=torch.ones(count),
        mask_validity=torch.ones(count),
        solver_support=torch.zeros(count),
        per_query_support={},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=10,
            min_marginal_gain=0.0,
            main_dynamic_lookahead=5,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == list(range(10))
    assert result.metadata["main_dynamic_evaluation_count"] <= 60


def test_default_main_selection_uses_current_marginal_not_static_order():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 3.0],
                [3.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([100.0, 1.0, 1.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([0.0, 1.0, 0.5]),
        per_query_support={"hard": {1: 10.0, 2: 3.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            target_query_support=10.0,
            kc_weight=10.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            main_dynamic_lookahead=0,
        ),
    )

    assert result.sampled_idx.tolist() == [1]
    assert result.metadata["main_dynamic_lookahead"] == 0


def test_full_gaussian_set_selection_keeps_source_anchor_before_solver_augmentation():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [5.0, 0.0, 3.0],
                [8.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 0.0, 0.1, 0.1]),
        visibility_score=torch.ones(4),
        mask_validity=torch.ones(4),
        solver_support=torch.tensor([0.0, 0.0, 10.0, 9.0]),
        per_query_support={"q": {2: 2.0, 3: 2.0}},
        source_anchor_idx=torch.tensor([0, 1], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=1.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 1, 2]
    assert result.metadata["source_anchor_count"] == 2
    assert result.metadata["source_anchor_selected_count"] == 2


def test_source_anchor_can_be_forced_past_visibility_and_mask_gates():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [5.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 0.0, 10.0]),
        visibility_score=torch.tensor([0.0, 0.0, 1.0]),
        mask_validity=torch.tensor([0.0, 0.0, 1.0]),
        solver_support=torch.tensor([0.0, 0.0, 1.0]),
        per_query_support={},
        source_anchor_idx=torch.tensor([0, 1], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            min_visibility=1.0,
            min_mask_validity=1.0,
            force_source_anchor=True,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist()[:2] == [0, 1]
    assert result.metadata["source_anchor_selected_count"] == 2
    assert result.metadata["force_source_anchor"] is True


def test_soft_source_anchor_prior_does_not_force_unbacked_native_points():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [5.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 0.0, 0.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([0.0, 0.0, 1.0]),
        per_query_support={"hard": {2: 4.0}},
        source_anchor_idx=torch.tensor([0], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            source_anchor_mode="soft",
            source_anchor_weight=0.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=1.0,
            query_support_weight=2.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [2]
    assert result.metadata["source_anchor_mode"] == "soft"
    assert result.metadata["source_anchor_selected_count"] == 0


def test_query_aware_candidate_pool_keeps_solver_backed_landmark_outside_static_topk():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [5.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([0.0, 0.0, 1.0]),
        per_query_support={"hard": {2: 4.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            candidate_pool_size=1,
            candidate_pool_query_top_k=1,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.5,
            geometry_weight=0.0,
            depth_weight=0.0,
            main_dynamic_lookahead=2,
        ),
    )

    assert result.sampled_idx.tolist() == [2]
    assert result.metadata["candidate_pool_static_count"] == 1
    assert result.metadata["candidate_pool_query_added_count"] == 1


def test_union_candidate_pool_keeps_component_topk_landmarks_outside_static_topk():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [5.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        visibility_score=torch.tensor([1.0, 10.0, 1.0, 1.0]),
        mask_validity=torch.tensor([1.0, 1.0, 10.0, 1.0]),
        solver_support=torch.tensor([0.0, 0.0, 0.0, 10.0]),
        per_query_support={"hard": {3: 4.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            candidate_pool_size=1,
            candidate_pool_kc_top_k=1,
            candidate_pool_visibility_top_k=1,
            candidate_pool_mask_top_k=1,
            candidate_pool_solver_top_k=1,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.5,
            geometry_weight=0.0,
            depth_weight=0.0,
            main_dynamic_lookahead=4,
        ),
    )

    assert result.sampled_idx.tolist() == [3]
    assert result.metadata["candidate_pool_static_count"] == 1
    assert result.metadata["candidate_pool_kc_added_count"] == 0
    assert result.metadata["candidate_pool_visibility_added_count"] == 1
    assert result.metadata["candidate_pool_mask_added_count"] == 1
    assert result.metadata["candidate_pool_solver_added_count"] == 1
    assert result.metadata["candidate_count"] == 4


def test_query_pnp_bearing_gain_prefers_non_redundant_solver_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [5.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 1.0}},
        per_query_observations={
            "hard": {
                0: {"cell": [0, 0], "depth_bin": 0, "bearing": [0.0, 0.0, 1.0], "camera_xyz": [0.0, 0.0, 1.0]},
                1: {"cell": [0, 0], "depth_bin": 0, "bearing": [0.0, 0.01, 1.0], "camera_xyz": [0.1, 0.0, 1.0]},
                2: {"cell": [0, 0], "depth_bin": 0, "bearing": [1.0, 0.0, 1.0], "camera_xyz": [5.0, 0.0, 3.0]},
            }
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=20.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            query_bearing_weight=30.0,
            query_pnp_geometry_weight=10.0,
            main_dynamic_lookahead=3,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["query_bearing_weight"] == 30.0
    assert result.metadata["query_pnp_geometry_weight"] == 10.0


def test_descriptor_conflict_graph_penalizes_redundant_far_landmark_pairs():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [5.0, 0.0, 1.0],
                [0.0, 5.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 2.0}},
        descriptor_conflict_edges={0: {1: 5.0}, 1: {0: 5.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=20.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            descriptor_conflict_weight=5.0,
            main_dynamic_lookahead=3,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["descriptor_conflict_edge_count"] == 2
    assert result.metadata["descriptor_conflict_weight"] == 5.0


def test_auto_descriptor_conflict_graph_blocks_far_descriptor_duplicates_without_source_anchor():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [8.0, 0.0, 1.0],
                [0.0, 5.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 2.0}},
        descriptor_features=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9999, 0.01, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            target_query_support=20.0,
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            descriptor_conflict_weight=10.0,
            auto_descriptor_conflict_top_k=1,
            auto_descriptor_conflict_min_cosine=0.99,
            auto_descriptor_conflict_min_spatial_distance_m=2.0,
            main_dynamic_lookahead=3,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["auto_descriptor_conflict_edge_count"] == 2
    assert result.metadata["descriptor_conflict_edge_count"] == 2


def test_sparse_validation_risk_penalizes_landmarks_that_regress_selfmap_pnp():
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    signals = SparseSetSignals(
        xyz=xyz,
        kc_score=torch.tensor([1.0, 0.9, 0.8]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([1.0, 0.95, 0.1]),
        sparse_validation_risk=torch.tensor([10.0, 0.0, 0.0]),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=-10.0,
            kc_weight=1.0,
            solver_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            sparse_validation_risk_weight=5.0,
            source_anchor_mode="off",
        ),
    )

    assert result.sampled_idx.tolist() == [1]
    assert result.metadata["sparse_validation_risk_weight"] == 5.0
    assert result.metadata["sparse_validation_risk_nonzero_count"] == 1


def test_sparse_validation_risk_hard_reject_removes_regression_basin_candidates():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([1.0, 10.0, 5.0], dtype=torch.float32),
        visibility_score=torch.ones(3, dtype=torch.float32),
        mask_validity=torch.ones(3, dtype=torch.float32),
        solver_support=torch.ones(3, dtype=torch.float32),
        per_query_support={"q": {0: 1.0, 1: 100.0, 2: 2.0}},
        sparse_validation_risk=torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            sparse_validation_risk_reject_threshold=0.1,
            source_anchor_mode="off",
        ),
    )

    assert 1 not in set(result.sampled_idx.tolist())
    assert result.metadata["sparse_validation_risk_hard_rejected_count"] == 1


def test_sparse_validation_risk_expands_to_far_descriptor_substitutes_before_rejecting():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [8.0, 0.0, 1.0],
                [0.0, 5.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([9.0, 8.0, 1.0], dtype=torch.float32),
        visibility_score=torch.ones(3, dtype=torch.float32),
        mask_validity=torch.ones(3, dtype=torch.float32),
        solver_support=torch.tensor([9.0, 8.0, 1.0], dtype=torch.float32),
        per_query_support={"q": {0: 9.0, 1: 8.0, 2: 1.0}},
        sparse_validation_risk=torch.tensor([5.0, 0.0, 0.0], dtype=torch.float32),
        descriptor_features=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9999, 0.01, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            sparse_validation_risk_reject_threshold=0.1,
            sparse_validation_risk_expand_top_k=1,
            sparse_validation_risk_expand_neighbors_per_seed=1,
            sparse_validation_risk_expand_candidate_limit=3,
            sparse_validation_risk_expand_min_cosine=0.99,
            sparse_validation_risk_expand_min_spatial_distance_m=2.0,
            source_anchor_mode="off",
        ),
    )

    assert result.sampled_idx.tolist() == [2]
    assert result.metadata["sparse_validation_risk_expanded_count"] == 1
    assert result.metadata["sparse_validation_risk_hard_rejected_count"] == 2


def test_local_search_adds_query_validation_support_without_source_or_kc_anchor():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.1],
                [4.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.1]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([1.0, 1.0, 0.1]),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 1.0}},
        per_query_observations={
            "hard": {
                0: {"cell": [0, 0], "depth_bin": 0, "bearing": [0.0, 0.0, 1.0], "camera_xyz": [0.0, 0.0, 1.0]},
                1: {"cell": [0, 0], "depth_bin": 0, "bearing": [0.01, 0.0, 1.0], "camera_xyz": [0.1, 0.0, 1.1]},
                2: {"cell": [3, 0], "depth_bin": 3, "bearing": [1.0, 0.0, 1.0], "camera_xyz": [4.0, 0.0, 5.0]},
            }
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=8.0,
            min_landmarks=2,
            source_anchor_mode="off",
            kc_anchor_count=0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            validation_min_query_cells=2,
            validation_min_query_depth_bins=2,
            validation_min_query_bearing_spread=0.25,
            local_search_rounds=2,
            local_search_candidates_per_query=3,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 1, 2]
    assert result.metadata["local_search_added_count"] == 1
    assert result.metadata["validation_failed_query_count"] == 0
    assert result.metadata["final_query_cell_counts"]["hard"] == 2
    assert result.metadata["final_query_depth_bin_counts"]["hard"] == 2


def test_local_search_reserve_keeps_budget_for_validation_repair():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.1],
                [4.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 9.0, 0.1]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 9.0, 2: 1.0}},
        per_query_observations={
            "hard": {
                0: {"cell": [0, 0], "depth_bin": 0, "bearing": [0.0, 0.0, 1.0], "camera_xyz": [0.0, 0.0, 1.0]},
                1: {"cell": [0, 0], "depth_bin": 0, "bearing": [0.01, 0.0, 1.0], "camera_xyz": [0.1, 0.0, 1.1]},
                2: {"cell": [3, 0], "depth_bin": 3, "bearing": [1.0, 0.0, 1.0], "camera_xyz": [4.0, 0.0, 5.0]},
            }
        },
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=2,
            min_marginal_gain=0.0,
            source_anchor_mode="off",
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            validation_min_query_cells=2,
            validation_min_query_depth_bins=2,
            validation_min_query_bearing_spread=0.25,
            local_search_rounds=1,
            local_search_candidates_per_query=3,
            local_search_max_additions=1,
            local_search_reserve_count=1,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["main_selection_cap"] == 1
    assert result.metadata["local_search_added_count"] == 1
    assert result.metadata["validation_failed_query_count"] == 0


def test_local_pruning_removes_conflicted_point_when_query_validation_stays_satisfied():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [8.0, 0.0, 1.0],
                [0.0, 5.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.zeros(3),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={"hard": {0: 10.0, 1: 1.0, 2: 2.0}},
        per_query_observations={
            "hard": {
                0: {"cell": [0, 0], "depth_bin": 0},
                1: {"cell": [0, 0], "depth_bin": 0},
                2: {"cell": [3, 0], "depth_bin": 3},
            }
        },
        descriptor_conflict_edges={0: {1: 5.0}, 1: {0: 5.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            source_anchor_mode="off",
            kc_weight=0.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            descriptor_conflict_weight=0.0,
            validation_min_query_cells=2,
            validation_min_query_depth_bins=2,
            local_prune_conflict_threshold=1.0,
            local_prune_validate_queries=True,
            final_prune_min_keep_count=2,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["local_pruned_conflict_count"] == 1
    assert result.metadata["validation_failed_query_count"] == 0


def test_final_pruning_removes_selected_landmarks_without_query_utility():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [5.0, 0.0, 2.0],
                [0.0, 5.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([10.0, 0.0, 0.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.zeros(3),
        per_query_support={"hard": {1: 5.0, 2: 5.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            final_prune_no_query_utility=True,
            final_prune_min_keep_count=1,
            main_dynamic_lookahead=3,
        ),
    )

    assert result.sampled_idx.tolist() == [1, 2]
    assert result.metadata["final_pruned_no_query_utility_count"] == 1
    assert result.metadata["selected_count"] == 2


def test_final_conflict_pruning_preserves_sparse_validation_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 10.0, 9.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.zeros(3),
        per_query_support={"main": {1: 1.0, 2: 1.0}},
        validation_per_query_support={"protected": {0: 0.1}},
        descriptor_conflict_edges={0: {1: 5.0}, 1: {0: 5.0}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            validation_query_prefill_target_fraction=1.0,
            validation_query_prefill_force_all=True,
            final_prune_conflict_threshold=1.0,
            final_prune_min_keep_count=2,
        ),
    )

    assert 0 in result.sampled_idx.tolist()
    assert result.metadata["final_pruned_conflict_count"] == 1


def test_nonreference_competition_pruning_preserves_sparse_validation_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 9.0, 10.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.zeros(3),
        per_query_support={"main": {1: 1.0, 2: 1.0}},
        validation_per_query_support={"protected": {0: 0.1}},
        per_query_match_competition_risk={"q": {0: 10.0, 1: 5.0}},
        per_query_match_competition_reference_landmarks={"q": {2}},
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
            validation_query_prefill_target_fraction=1.0,
            validation_query_prefill_force_all=True,
            final_prune_nonreference_query_match_competition=True,
            final_prune_min_keep_count=2,
        ),
    )

    assert 0 in result.sampled_idx.tolist()
    assert result.metadata["final_pruned_nonreference_query_match_competition_count"] == 1


def test_non_source_augmentation_can_require_solver_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 10.0, 1.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([0.0, 0.0, 0.6]),
        per_query_support={},
        source_anchor_idx=torch.tensor([0], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            augmentation_min_solver_support=0.5,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["augmentation_solver_gate_rejected_count"] == 1
    assert result.metadata["augmentation_min_solver_support"] == 0.5


def test_non_source_augmentation_can_require_keypoint_consensus_support():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.0, 0.2, 0.8]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.ones(3),
        per_query_support={},
        source_anchor_idx=torch.tensor([0], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=3,
            min_marginal_gain=0.0,
            augmentation_min_kc_score=0.5,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=0.0,
            query_support_weight=0.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0, 2]
    assert result.metadata["augmentation_kc_gate_rejected_count"] == 1
    assert result.metadata["augmentation_min_kc_score"] == 0.5


def test_source_query_prefill_protects_query_backed_soft_source_anchor():
    signals = SparseSetSignals(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        kc_score=torch.tensor([0.1, 0.0, 10.0]),
        visibility_score=torch.ones(3),
        mask_validity=torch.ones(3),
        solver_support=torch.tensor([0.1, 0.0, 1.0]),
        per_query_support={"easy": {0: 8.0, 2: 100.0}},
        source_anchor_idx=torch.tensor([0, 1], dtype=torch.long),
    )

    result = select_sparse_solver_set_from_full_gaussians(
        signals,
        SparseSetSelectionConfig(
            max_landmarks=1,
            min_marginal_gain=0.0,
            target_query_support=8.0,
            source_anchor_mode="soft",
            source_anchor_weight=0.0,
            source_query_prefill_target_fraction=1.0,
            kc_weight=1.0,
            visibility_weight=0.0,
            mask_weight=0.0,
            solver_weight=1.0,
            query_support_weight=1.0,
            geometry_weight=0.0,
            depth_weight=0.0,
        ),
    )

    assert result.sampled_idx.tolist() == [0]
    assert result.metadata["source_query_prefill_selected_count"] == 1
