from pathlib import Path

import pytest

from loc_gs.scripts.build_ulfloc_scene_matcher_pair_cache import (
    _repeat_row_source_view_ids,
    build_argparser,
    validate_split_name,
)


def test_scene_matcher_pair_cache_parser_accepts_required_options():
    args = build_argparser().parse_args(
        [
            "--source_path",
            "/src",
            "--model_path",
            "/model",
            "--input_log_dir",
            "/model/log",
            "--cfg",
            "/model/log/config.yaml",
            "--output_dir",
            "/out",
            "--split_name",
            "selfmap_train",
            "--topk",
            "4",
            "--max_queries",
            "8",
            "--camera_split",
            "train",
        ]
    )

    assert args.source_path == Path("/src")
    assert args.topk == 4
    assert args.max_queries == 8
    assert args.camera_split == "train"


def test_scene_matcher_pair_cache_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        validate_split_name("test")


def test_scene_matcher_pair_cache_repeats_source_view_ids_per_keypoint_row():
    row_ids = _repeat_row_source_view_ids(
        [
            ("seq1/frame00001.png", 2),
            ("seq1/frame00002.png", 3),
        ]
    )

    assert row_ids == [
        "seq1/frame00001.png",
        "seq1/frame00001.png",
        "seq1/frame00002.png",
        "seq1/frame00002.png",
        "seq1/frame00002.png",
    ]
