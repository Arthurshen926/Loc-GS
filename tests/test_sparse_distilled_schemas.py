import pytest

from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.teacher.labels import TeacherLabelBatch, build_teacher_labels


def test_sparse_candidate_batch_requires_matching_topk_shapes():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train_selfmap",
        query_id="q1",
        keypoint_xy=[[10.0, 20.0], [30.0, 40.0]],
        candidate_landmark_ids=[[1, 2], [3]],
        candidate_scores=[[0.9, 0.8], [0.7, 0.6]],
    )

    with pytest.raises(ValueError, match="same top-k length"):
        batch.validate()


def test_teacher_labels_reject_dense_teacher_on_test_split():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="test",
        query_id="q1",
        keypoint_xy=[[10.0, 20.0]],
        candidate_landmark_ids=[[1, 2]],
        candidate_scores=[[0.9, 0.8]],
    )

    with pytest.raises(ValueError, match="test split"):
        build_teacher_labels(batch, geometric_correct=[[True, False]], dense_consistent=[[True, False]])


def test_teacher_labels_preserve_per_candidate_roles():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train_selfmap",
        query_id="q1",
        keypoint_xy=[[10.0, 20.0]],
        candidate_landmark_ids=[[1, 2]],
        candidate_scores=[[0.9, 0.8]],
    )

    labels = build_teacher_labels(
        batch,
        geometric_correct=[[True, False]],
        dense_consistent=[[True, False]],
        sparse_inlier=[[True, False]],
        reprojection_error_px=[[1.0, 18.0]],
    )

    assert isinstance(labels, TeacherLabelBatch)
    assert labels.label_roles == [["protected_support", "hard_negative"]]
    assert labels.positive_count == 1
    assert labels.hard_negative_count == 1
