import pytest
import torch

from loc_gs.stdloc_native.sparse_set_composition_audit import audit_sparse_set_composition


def test_sparse_set_composition_audit_reports_overlap_support_risk_and_scores():
    profile = {
        "schema": "loc_gs_sparse_pnp_validation_profile_v2",
        "split_name": "train_dev",
        "attribution_status": "attributed",
        "landmark_regression_risk": {"4": 3.0, "6": 2.0},
        "protected_per_query_support": {
            "q_good": {"0": 2.0, "2": 3.0},
            "q_good2": {"5": 1.0},
        },
        "validated_per_query_support": {
            "q_hard": {"4": 1.0, "5": 4.0},
        },
    }

    report = audit_sparse_set_composition(
        selected_idx=torch.tensor([2, 3, 4, 5]),
        baseline_idx=torch.tensor([0, 1, 2, 3]),
        sparse_validation_profile=profile,
        score_vectors={
            "kc": torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.6, 0.0]),
            "visibility": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 0.0]),
        },
    )

    assert report["selected_count"] == 4
    assert report["baseline_overlap_count"] == 2
    assert report["baseline_overlap_fraction"] == pytest.approx(0.5)
    assert report["protected_support"]["landmark_selected_count"] == 2
    assert report["protected_support"]["landmark_missing_count"] == 1
    assert report["protected_support"]["support_mass_selected_fraction"] == pytest.approx(4.0 / 6.0)
    assert report["validated_support"]["landmark_selected_count"] == 2
    assert report["regression_risk"]["selected_count"] == 1
    assert report["regression_risk"]["selected_mass"] == pytest.approx(3.0)
    assert report["score_stats"]["kc"]["selected_mean"] == pytest.approx((0.7 + 0.2 + 0.1 + 0.6) / 4.0)
    assert report["score_stats"]["kc"]["missing_baseline_mean"] == pytest.approx((0.9 + 0.8) / 2.0)
    assert report["per_query_support"]["protected:q_good"]["selected_landmark_count"] == 1
    assert report["per_query_support"]["protected:q_good"]["selected_support_fraction"] == pytest.approx(3.0 / 5.0)
