import numpy as np

from loc_gs.dense_support.patch_dense_candidates import (
    PatchDenseCandidatePolicy,
    generate_patch_dense_candidates,
)


def _one_hot_feature_map(height=6, width=6):
    features = np.zeros((height * width, height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            features[y * width + x, y, x] = 1.0
    return features


def test_patch_dense_candidates_emit_patch_scores_and_do_not_select_pose():
    query = _one_hot_feature_map()
    rendered = query.copy()
    depth = np.full((6, 6), 10.0, dtype=np.float32)
    anchors = {
        "query_xy": np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float32),
        "render_xy": np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float32),
    }

    result = generate_patch_dense_candidates(
        query,
        rendered,
        depth,
        anchors,
        policy=PatchDenseCandidatePolicy(grid_rows=2, grid_cols=2, sample_stride=1, top_patch_count=4),
    )

    assert result["schema"] == "loc_gs_patch_dense_candidates_v1"
    assert result["diagnostic_only"] is True
    assert result["does_not_select_final_pose"] is True
    assert result["uses_gt"] is False
    assert result["match_count"] > 0
    assert result["xy"].shape[1] == 2
    assert result["render_xy"].shape[1] == 2
    assert result["depth"].shape[0] == result["xy"].shape[0]
    assert {"local_match_quality", "anchor_consistency", "off_patch_consistency", "ambiguity_score"} <= set(
        result["patches"][0]
    )


def test_patch_dense_candidates_downweight_ambiguous_local_repeated_patch():
    query = _one_hot_feature_map()
    rendered = query.copy()
    rendered[:, 0, 1] = rendered[:, 0, 0]
    depth = np.full((6, 6), 10.0, dtype=np.float32)
    anchors = {
        "query_xy": np.array([[4.0, 4.0]], dtype=np.float32),
        "render_xy": np.array([[4.0, 4.0]], dtype=np.float32),
    }

    result = generate_patch_dense_candidates(
        query,
        rendered,
        depth,
        anchors,
        policy=PatchDenseCandidatePolicy(grid_rows=2, grid_cols=2, sample_stride=1, top_patch_count=4),
    )

    ambiguous_patch = max(result["patches"], key=lambda row: row["ambiguity_score"])
    stable_patch = min(result["patches"], key=lambda row: row["ambiguity_score"])
    assert ambiguous_patch["ambiguity_score"] > stable_patch["ambiguity_score"]
    assert ambiguous_patch["patch_weight"] < stable_patch["patch_weight"]


def test_patch_dense_candidates_anchor_and_off_patch_consistency_follow_anchor_offset():
    query = _one_hot_feature_map(height=5, width=5)
    rendered = np.zeros_like(query)
    rendered[:, :, 1:] = query[:, :, :-1]
    depth = np.full((5, 5), 4.0, dtype=np.float32)
    anchors = {
        "query_xy": np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        "render_xy": np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype=np.float32),
    }

    result = generate_patch_dense_candidates(
        query,
        rendered,
        depth,
        anchors,
        policy=PatchDenseCandidatePolicy(grid_rows=1, grid_cols=1, sample_stride=1, top_patch_count=1),
    )

    patch = result["patches"][0]
    assert patch["anchor_consistency"] > 0.8
    assert patch["off_patch_consistency"] > 0.8
    assert patch["median_offset_xy"] == [1.0, 0.0]


def test_patch_dense_candidates_can_lift_render_depth_to_world_points():
    query = _one_hot_feature_map(height=4, width=4)
    rendered = query.copy()
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    intrinsic = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    render_pose = np.eye(4, dtype=np.float32)

    result = generate_patch_dense_candidates(
        query,
        rendered,
        depth,
        sparse_anchors=None,
        policy=PatchDenseCandidatePolicy(grid_rows=1, grid_cols=1, sample_stride=2, top_patch_count=1),
        render_pose_w2c=render_pose,
        intrinsic=intrinsic,
    )

    assert "p3d" in result
    assert result["p3d"].shape == (result["match_count"], 3)
    assert np.allclose(result["p3d"][:, 2], 2.0)
