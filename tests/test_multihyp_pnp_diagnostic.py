import pytest

from loc_gs.stdloc_native.multihypothesis_pnp import build_multihypothesis_diagnostic_plan


def test_multihypothesis_pnp_requires_diagnostic_flag():
    with pytest.raises(ValueError, match="diagnostic"):
        build_multihypothesis_diagnostic_plan(
            matches=[{"group": "a", "score": 1.0}],
            diagnostic=False,
        )


def test_multihypothesis_pnp_groups_hypotheses_and_never_marks_main_result():
    result = build_multihypothesis_diagnostic_plan(
        matches=[
            {"group": "bank0", "score": 1.0, "inlier_prior": 10, "logdet_h": 2.0, "dense_verifier": 0.5},
            {"group": "bank1", "score": 1.0, "inlier_prior": 5, "logdet_h": 1.0, "dense_verifier": 0.2},
            {"group": "bank0", "score": 0.5, "inlier_prior": 4, "logdet_h": 1.5, "dense_verifier": 0.7},
        ],
        diagnostic=True,
    )

    assert result["metadata"]["paper_safe_role"] == "diagnostic_ablation_only"
    assert result["metadata"]["main_method_allowed"] is False
    assert result["hypotheses"][0]["group"] == "bank0"
    assert len(result["hypotheses"]) == 2
