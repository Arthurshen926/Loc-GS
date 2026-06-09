from __future__ import annotations

import json
from pathlib import Path

import pytest

from loc_gs.feedback.io import save_feedback_bank
from loc_gs.scripts.build_correspondence_supervision import build_argparser, main


def test_build_correspondence_supervision_cli_writes_targets_and_audit(tmp_path: Path) -> None:
    bank = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank,
        [
            {
                "scene": "ShopFacade",
                "query_id": "q0",
                "matched_gaussian_id": "1",
                "keypoint_xy": [10.0, 12.0],
                "descriptor_score": 0.9,
                "descriptor_margin": 0.4,
                "pnp_inlier": True,
                "reprojection_error_px": 0.5,
            },
            {
                "scene": "ShopFacade",
                "query_id": "q0",
                "matched_gaussian_id": "2",
                "keypoint_xy": [11.0, 12.0],
                "descriptor_score": 0.88,
                "descriptor_margin": 0.01,
                "pnp_inlier": False,
                "reprojection_error_px": 30.0,
            },
        ],
        {"scene": "ShopFacade", "split_name": "selfmap_train"},
    )
    output_dir = tmp_path / "out"
    args = build_argparser().parse_args(["--feedback_bank", str(bank), "--output_dir", str(output_dir)])

    assert main(args) == 0

    targets = json.loads((output_dir / "targets.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))

    assert targets["match_scorer"]["labels"] == [1, 0]
    assert metrics["record_count"] == 2
    assert metrics["positive_count"] == 1
    assert split_audit["test_split_used"] is False
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_correspondence_supervision_cli_uses_sparse_validation_profile(tmp_path: Path) -> None:
    bank = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank,
        [
            {
                "scene": "ShopFacade",
                "query_id": "q_regress",
                "matched_gaussian_id": "1",
                "keypoint_xy": [10.0, 12.0],
                "descriptor_score": 0.9,
                "descriptor_margin": 0.4,
                "pnp_inlier": True,
                "reprojection_error_px": 0.5,
            },
            {
                "scene": "ShopFacade",
                "query_id": "q_regress",
                "matched_gaussian_id": "2",
                "keypoint_xy": [11.0, 12.0],
                "descriptor_score": 0.88,
                "descriptor_margin": 0.01,
                "pnp_inlier": False,
                "reprojection_error_px": 30.0,
            },
        ],
        {"scene": "ShopFacade", "split_name": "selfmap_train"},
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "loc_gs_sparse_pnp_validation_profile_v1",
                "split_name": "train_dev",
                "query_regression_delta_cm": {"q_regress": 50.0},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out_profile"
    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--output_dir",
            str(output_dir),
            "--sparse_validation_profile",
            str(profile),
        ]
    )

    assert main(args) == 0

    artifact = json.loads((output_dir / "correspondence_supervision.json").read_text(encoding="utf-8"))
    targets = json.loads((output_dir / "targets.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert artifact["hard_negative_count"] == 1
    assert targets["match_scorer"]["hard_negative"] == [False, True]
    assert metrics["hard_negative_count"] == 1


def test_build_correspondence_supervision_cli_rejects_test_split(tmp_path: Path) -> None:
    bank = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(bank, [], {"scene": "ShopFacade", "split_name": "test"})
    args = build_argparser().parse_args(["--feedback_bank", str(bank), "--output_dir", str(tmp_path / "out")])

    with pytest.raises(ValueError, match="test split"):
        main(args)
