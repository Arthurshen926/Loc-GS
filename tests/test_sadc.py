import numpy as np

from loc_gs.dense_support.sadc import (
    SADCCorrespondencePolicy,
    apply_sadc_anchor_monotonic_update,
    decide_sadc_activation,
    filter_sadc_correspondences,
    sanitize_sadc_correspondences,
    score_sadc_correspondences,
)


def _camera():
    pose = np.eye(4, dtype=np.float32)
    intrinsic = np.array(
        [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return pose, intrinsic


def test_sadc_scores_anchor_flow_consistent_matches_higher_than_flow_outliers():
    sparse_pose, intrinsic = _camera()
    sparse_anchors = {
        "query_xy": np.array([[42.0, 50.0], [62.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[40.0, 50.0], [60.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
        "inliers": np.array([0, 1], dtype=np.int32),
    }
    candidates = {
        "query_xy": np.array([[52.0, 50.0], [80.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "match_scores": np.array([0.9, 0.9], dtype=np.float32),
    }

    scored = score_sadc_correspondences(
        candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SADCCorrespondencePolicy(anchor_flow_scale_px=4.0),
    )

    assert scored["schema"] == "loc_gs_sadc_correspondence_scores_v1"
    assert scored["uses_gt"] is False
    assert scored["core_signals"] == ["match_quality", "anchor_flow_consistency", "geometry"]
    assert scored["weights"][0] > scored["weights"][1]
    assert scored["anchor_flow_consistency"][0] > 0.9
    assert scored["anchor_flow_consistency"][1] < 0.01


def test_sadc_filter_keeps_high_weight_candidates_deterministically():
    sparse_pose, intrinsic = _camera()
    sparse_anchors = {
        "query_xy": np.array([[42.0, 50.0], [62.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[40.0, 50.0], [60.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 2.0]], dtype=np.float32),
        "inliers": np.array([0, 1], dtype=np.int32),
    }
    candidates = {
        "query_xy": np.array([[52.0, 50.0], [80.0, 50.0], [72.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[50.0, 50.0], [50.0, 50.0], [70.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.5], [0.2, 0.0, 2.0]], dtype=np.float32),
        "match_scores": np.array([0.9, 0.9, 0.8], dtype=np.float32),
        "sources": np.array(["native", "native", "patch"], dtype=object),
    }

    filtered = filter_sadc_correspondences(
        candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SADCCorrespondencePolicy(anchor_flow_scale_px=4.0, max_candidates=2),
    )

    assert filtered["schema"] == "loc_gs_sadc_filtered_correspondences_v1"
    assert filtered["query_xy"].shape == (2, 2)
    assert filtered["kept_count"] == 2
    assert filtered["dropped_count"] == 1
    assert filtered["query_xy"][0].tolist() == [52.0, 50.0]
    assert filtered["sources"].tolist() == ["native", "patch"]


def test_sadc_passthrough_preserves_native_candidate_order_and_count():
    sparse_pose, intrinsic = _camera()
    sparse_anchors = {
        "query_xy": np.array([[42.0, 50.0], [62.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[40.0, 50.0], [60.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 2.0]], dtype=np.float32),
        "inliers": np.array([0, 1], dtype=np.int32),
    }
    candidates = {
        "query_xy": np.array([[52.0, 50.0], [64.0, 50.0], [73.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[50.0, 50.0], [62.0, 50.0], [70.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0], [0.12, 0.0, 1.5], [0.2, 0.0, 2.0]], dtype=np.float32),
        "match_scores": np.array([0.9, 0.7, 0.6], dtype=np.float32),
        "sources": np.array(["native", "native", "native"], dtype=object),
    }

    out = sanitize_sadc_correspondences(
        candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SADCCorrespondencePolicy(mode="passthrough"),
    )

    assert out["mode"] == "passthrough"
    assert out["kept_count"] == 3
    assert out["dropped_count"] == 0
    np.testing.assert_allclose(out["query_xy"], candidates["query_xy"])
    np.testing.assert_allclose(out["render_xy"], candidates["render_xy"])
    np.testing.assert_allclose(out["p3d"], candidates["p3d"])
    np.testing.assert_allclose(out["match_scores"], candidates["match_scores"])
    assert out["kept_indices"].tolist() == [0, 1, 2]


def test_sadc_conflict_filter_preserves_native_core_except_extreme_conflicts():
    sparse_pose, intrinsic = _camera()
    sparse_anchors = {
        "query_xy": np.array([[42.0, 50.0], [62.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[40.0, 50.0], [60.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 2.0]], dtype=np.float32),
        "inliers": np.array([0, 1], dtype=np.int32),
    }
    candidates = {
        "query_xy": np.array([[52.0, 50.0], [64.0, 50.0], [95.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[50.0, 50.0], [62.0, 50.0], [70.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0], [0.12, 0.0, 1.5], [0.2, 0.0, 2.0]], dtype=np.float32),
        "match_scores": np.array([0.9, 0.7, 0.6], dtype=np.float32),
        "sources": np.array(["native", "native", "native"], dtype=object),
    }

    out = sanitize_sadc_correspondences(
        candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SADCCorrespondencePolicy(
            mode="conflict_filter",
            native_drop_percentile=66.0,
            min_native_keep_ratio=0.5,
        ),
    )

    assert out["mode"] == "conflict_filter"
    assert out["kept_indices"].tolist() == [0, 1]
    assert out["dropped_count"] == 1
    assert out["native_kept_count"] == 2
    assert out["patch_kept_count"] == 0


def test_sadc_patch_append_keeps_all_native_and_admits_only_high_confidence_patch():
    sparse_pose, intrinsic = _camera()
    sparse_anchors = {
        "query_xy": np.array([[42.0, 50.0], [62.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[40.0, 50.0], [60.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 2.0]], dtype=np.float32),
        "inliers": np.array([0, 1], dtype=np.int32),
    }
    candidates = {
        "query_xy": np.array([[52.0, 50.0], [64.0, 50.0], [72.0, 50.0], [96.0, 50.0]], dtype=np.float32),
        "render_xy": np.array([[50.0, 50.0], [62.0, 50.0], [70.0, 50.0], [70.0, 50.0]], dtype=np.float32),
        "p3d": np.array([[0.0, 0.0, 1.0], [0.12, 0.0, 1.5], [0.2, 0.0, 2.0], [0.2, 0.0, 2.0]], dtype=np.float32),
        "match_scores": np.array([0.8, 0.7, 0.95, 0.95], dtype=np.float32),
        "sources": np.array(["native", "native", "patch", "patch"], dtype=object),
    }

    out = sanitize_sadc_correspondences(
        candidates,
        sparse_pose=sparse_pose,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SADCCorrespondencePolicy(
            mode="patch_append",
            patch_add_percentile=50.0,
            max_patch_fraction=0.5,
        ),
    )

    assert out["mode"] == "patch_append"
    assert out["kept_indices"].tolist() == [0, 1, 2]
    assert out["native_kept_count"] == 2
    assert out["patch_kept_count"] == 1
    np.testing.assert_allclose(out["query_xy"][:2], candidates["query_xy"][:2])


def test_sadc_anchor_monotonic_shrinks_dense_update_that_breaks_sparse_anchors():
    sparse_pose, intrinsic = _camera()
    points = np.array(
        [[-0.1, 0.0, 2.0], [0.1, 0.0, 2.0], [0.0, 0.1, 2.5]],
        dtype=np.float32,
    )
    sparse_anchors = {
        "query_xy": np.array([[45.0, 50.0], [55.0, 50.0], [50.0, 54.0]], dtype=np.float32),
        "p3d": points,
        "inliers": np.arange(3, dtype=np.int32),
    }
    candidate = sparse_pose.copy()
    candidate[0, 3] = 0.5

    selected, diag = apply_sadc_anchor_monotonic_update(
        sparse_pose=sparse_pose,
        candidate_pose=candidate,
        sparse_anchors=sparse_anchors,
        intrinsic=intrinsic,
        image_size=(100, 100),
        policy=SADCCorrespondencePolicy(anchor_monotonic=True, anchor_monotonic_epsilon_px=0.1),
    )

    assert diag["schema"] == "loc_gs_sadc_anchor_monotonic_v1"
    assert diag["uses_gt"] is False
    assert diag["selected_alpha"] < 1.0
    assert diag["selected_anchor_median_px"] <= diag["sparse_anchor_median_px"] + 0.1
    assert np.linalg.norm(selected[:3, 3] - sparse_pose[:3, 3]) < np.linalg.norm(candidate[:3, 3] - sparse_pose[:3, 3])


def test_sadc_activation_keeps_native_default_when_dense_damage_risk_is_low():
    report = {"risk": 0.2, "uses_gt": False, "components": {"sparse_confidence": 1.0}}

    decision = decide_sadc_activation(
        report,
        SADCCorrespondencePolicy(
            activation_mode="dense_damage_risk",
            activation_min_risk=0.5,
            activation_min_sparse_confidence=0.25,
        ),
    )

    assert decision["schema"] == "loc_gs_sadc_activation_v1"
    assert decision["uses_gt"] is False
    assert decision["active"] is False
    assert decision["mode"] == "dense_damage_risk"
    assert decision["reason"] == "risk_below_threshold"


def test_sadc_activation_turns_on_for_high_risk_observable_dense_damage():
    report = {"risk": 0.8, "uses_gt": False, "components": {"sparse_confidence": 0.75}}

    decision = decide_sadc_activation(
        report,
        SADCCorrespondencePolicy(
            activation_mode="dense_damage_risk",
            activation_min_risk=0.5,
            activation_min_sparse_confidence=0.25,
        ),
    )

    assert decision["uses_gt"] is False
    assert decision["active"] is True
    assert decision["risk"] == 0.8
    assert decision["reason"] == "active_dense_damage_risk"
