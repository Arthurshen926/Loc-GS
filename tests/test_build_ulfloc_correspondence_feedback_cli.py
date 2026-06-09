from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch

from loc_gs.feedback.io import save_feedback_bank
from loc_gs.scripts.build_ulfloc_correspondence_feedback import build_argparser, main


def _write_bank(path: Path, *, split_name: str = "selfmap_train") -> None:
    save_feedback_bank(
        path,
        [
            {
                "scene": "ShopFacade",
                "query_id": "q0",
                "matched_gaussian_id": "2",
                "keypoint_xy": [10.0, 12.0],
                "query_xy_norm": [0.1, 0.2],
                "descriptor_score": 0.9,
                "descriptor_margin": 0.4,
                "pnp_inlier": True,
                "reprojection_error_px": 0.5,
            },
            {
                "scene": "ShopFacade",
                "query_id": "q0",
                "matched_gaussian_id": "4",
                "keypoint_xy": [11.0, 12.0],
                "query_xy_norm": [0.11, 0.2],
                "descriptor_score": 0.88,
                "descriptor_margin": 0.01,
                "pnp_inlier": False,
                "reprojection_error_px": 20.0,
            },
        ],
        {"scene": "ShopFacade", "split_name": split_name},
    )


def test_build_ulfloc_correspondence_feedback_cli_writes_pickle_and_audit(tmp_path: Path) -> None:
    bank = tmp_path / "feedback_bank.jsonl"
    _write_bank(bank)
    output_dir = tmp_path / "out"
    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--output_dir",
            str(output_dir),
            "--num_landmarks",
            "8",
        ]
    )

    assert main(args) == 0

    with (output_dir / "solver_feedback.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))

    weights = torch.as_tensor(payload["landmark_weights"], dtype=torch.float32)
    assert weights[2] > 1.0
    assert weights[4] < 1.0
    assert metrics["positive_count"] == 1
    assert metrics["conflict_edge_count"] == 1
    assert split_audit["test_split_used"] is False
    assert (output_dir / "correspondence_targets.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_ulfloc_correspondence_feedback_cli_rejects_test_split(tmp_path: Path) -> None:
    bank = tmp_path / "feedback_bank.jsonl"
    _write_bank(bank, split_name="test")
    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--output_dir",
            str(tmp_path / "out"),
            "--num_landmarks",
            "8",
        ]
    )

    with pytest.raises(ValueError, match="test split"):
        main(args)

