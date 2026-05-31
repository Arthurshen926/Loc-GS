from loc_gs.diagnostics.apd_dense_damage_risk import (
    compute_dense_damage_risk,
    evaluate_dense_damage_risk,
    label_dense_damage_from_gt,
)


def _row(**kwargs):
    base = {
        "sparse_inlier_count": 96,
        "sparse_pose_anchor_median_px": 1.5,
        "base_dense_pose_anchor_median_px": 1.6,
        "base_dense_vs_sparse_translation_delta_m": 0.02,
        "base_dense_vs_sparse_rotation_delta_deg": 0.05,
        "base_dense_pose_p90_reprojection_error_px": 2.0,
        "sparse_te_cm": 8.0,
        "base_dense_te_cm": 9.0,
    }
    base.update(kwargs)
    return base


def test_dense_damage_risk_uses_only_observable_conflict_signals():
    safe = _row()
    risky = _row(
        base_dense_pose_anchor_median_px=9.0,
        base_dense_vs_sparse_translation_delta_m=0.45,
        base_dense_vs_sparse_rotation_delta_deg=4.0,
        base_dense_pose_p90_reprojection_error_px=12.0,
    )

    safe_score = compute_dense_damage_risk(safe)
    risky_score = compute_dense_damage_risk(risky)

    assert risky_score["uses_gt"] is False
    assert risky_score["risk"] > safe_score["risk"]
    assert risky_score["components"]["sparse_confidence"] > 0.0
    assert risky_score["components"]["anchor_conflict"] > safe_score["components"]["anchor_conflict"]


def test_dense_damage_risk_treats_large_dense_update_as_risky_even_without_anchor_median_conflict():
    score = compute_dense_damage_risk(
        _row(
            sparse_inlier_count=160,
            base_dense_pose_anchor_median_px=1.6,
            base_dense_vs_sparse_translation_delta_m=0.80,
            base_dense_vs_sparse_rotation_delta_deg=8.0,
            base_dense_pose_p90_reprojection_error_px=20.0,
        )
    )

    assert score["risk"] > 0.75


def test_dense_damage_label_is_gt_validation_only():
    assert label_dense_damage_from_gt(_row(sparse_te_cm=10.0, base_dense_te_cm=35.0)) is True
    assert label_dense_damage_from_gt(_row(sparse_te_cm=40.0, base_dense_te_cm=80.0)) is False
    assert label_dense_damage_from_gt(_row(sparse_te_cm=10.0, base_dense_te_cm=20.0)) is False


def test_evaluate_dense_damage_risk_reports_auc_and_topk_precision():
    rows = [
        _row(sparse_te_cm=7.0, base_dense_te_cm=8.0),
        _row(sparse_te_cm=8.0, base_dense_te_cm=40.0, base_dense_pose_anchor_median_px=9.0, base_dense_vs_sparse_translation_delta_m=0.4),
        _row(sparse_te_cm=7.0, base_dense_te_cm=8.0),
        _row(sparse_te_cm=9.0, base_dense_te_cm=36.0, base_dense_pose_anchor_median_px=8.0, base_dense_vs_sparse_translation_delta_m=0.3),
    ]

    report = evaluate_dense_damage_risk(rows)

    assert report["schema"] == "loc_gs_apd_dense_damage_risk_eval_v1"
    assert report["uses_gt_for_scoring"] is False
    assert report["positive_count"] == 2
    assert report["auc"] == 1.0
    assert report["precision_at_positive_count"] == 1.0
    assert report["recall_at_positive_count"] == 1.0
