import numpy as np

from loc_gs.scripts.mine_sparse_base_confidence import (
    build_argparser,
    classify_sparse_failure,
    summarize_sparse_confidence,
)


def test_sparse_confidence_summary_counts_matches_and_inliers():
    sparse = {
        "query_xy": np.zeros((10, 2), dtype=np.float32),
        "inliers": np.array([0, 2, 4], dtype=np.int64),
    }

    summary = summarize_sparse_confidence(sparse)

    assert summary["match_count"] == 10
    assert summary["inlier_count"] == 3
    assert summary["inlier_ratio"] == 0.3


def test_sparse_confidence_miner_accepts_train_split_options():
    args = build_argparser().parse_args(
        ["--scene", "KingsCollege", "--split", "train", "--max_queries", "12", "--low_inlier_threshold", "80"]
    )

    assert args.scene == "KingsCollege"
    assert args.split == "train"
    assert args.max_queries == 12
    assert args.low_inlier_threshold == 80


def test_sparse_failure_classification_detects_false_consensus():
    row = {
        "sparse_te_cm": 180.0,
        "gt_good_count": 30,
        "solver_inlier_gt_good_count": 2,
        "solver_inlier_gt_bad_count": 26,
        "inlier_count": 28,
    }

    assert classify_sparse_failure(row) == "false_consensus"


def test_sparse_failure_classification_detects_no_true_matches():
    row = {
        "sparse_te_cm": 180.0,
        "gt_good_count": 2,
        "solver_inlier_gt_good_count": 0,
        "solver_inlier_gt_bad_count": 6,
        "inlier_count": 6,
    }

    assert classify_sparse_failure(row) == "availability_or_descriptor_no_true_matches"
