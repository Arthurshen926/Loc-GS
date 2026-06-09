import pytest
import torch

from loc_gs.scripts.build_ulfloc_sparse_match_competition import (
    _positive_score_map,
    build_argparser,
    query_support_from_sparse_validation_profile,
)


def test_query_support_from_sparse_validation_profile_merges_protected_and_validated_support():
    profile = {
        "split_name": "train_dev_seed13_20p_validation",
        "protected_per_query_support": {
            "q.png": {"10": 2.0},
        },
        "validated_per_query_support": {
            "q.png": {"10": 3.0, "20": 4.0},
        },
    }

    support = query_support_from_sparse_validation_profile(profile)

    assert support == {"q.png": {10: 5.0, 20: 4.0}}


def test_query_support_from_sparse_validation_profile_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        query_support_from_sparse_validation_profile({"split_name": "test"})


def test_build_ulfloc_sparse_match_competition_parser_accepts_required_args():
    args = build_argparser().parse_args(
        [
            "--source_path",
            "/data/ShopFacade_train_dev",
            "--model_path",
            "/models/ShopFacade",
            "--input_log_dir",
            "/models/ShopFacade/log",
            "--cfg",
            "/models/ShopFacade/log/config.yaml",
            "--sparse_validation_profile",
            "/profiles/sparse_pnp_validation_profile.json",
            "--output_dir",
            "/out/competition",
            "--split_name",
            "train_dev_seed13_20p",
            "--query_id",
            "seq2/frame00049.png",
        ]
    )

    assert str(args.source_path) == "/data/ShopFacade_train_dev"
    assert str(args.model_path) == "/models/ShopFacade"
    assert str(args.input_log_dir) == "/models/ShopFacade/log"
    assert str(args.sparse_validation_profile) == "/profiles/sparse_pnp_validation_profile.json"
    assert args.query_id == ["seq2/frame00049.png"]


def test_positive_score_map_serializes_all_positive_scores_by_default():
    scores = torch.tensor([0.0, 2.5, -1.0, 1.0], dtype=torch.float32)

    assert _positive_score_map(scores) == {"1": 2.5, "3": 1.0}


def test_positive_score_map_can_limit_to_top_scores():
    scores = torch.tensor([0.0, 2.5, -1.0, 1.0], dtype=torch.float32)

    assert _positive_score_map(scores, limit=1) == {"1": 2.5}
