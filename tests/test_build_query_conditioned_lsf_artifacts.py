import json
import pickle

import pytest
import torch

from loc_gs.feedback.io import save_feedback_bank
from loc_gs.scripts.build_query_conditioned_lsf_artifacts import build_argparser, main


def _write_source_map(root, sampled_idx):
    detector = root / "detector"
    detector.mkdir(parents=True)
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor(sampled_idx, dtype=torch.long), handle)
    with (detector / "sampled_scores.pkl").open("wb") as handle:
        pickle.dump({"score_avg": torch.linspace(0.1, 0.9, 8)}, handle)


def _save_bank(path, *, split_name="selfmap_train"):
    save_feedback_bank(
        path,
        [
            {
                "scene": "ToyScene",
                "query_id": "query_000001",
                "matched_gaussian_id": "1",
                "matched_landmark_id": "1",
                "descriptor_score": 0.95,
                "match_rank": 1,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "pnp_success": True,
            },
            {
                "scene": "ToyScene",
                "query_id": "query_000001",
                "matched_gaussian_id": "4",
                "matched_landmark_id": "4",
                "descriptor_score": 0.92,
                "match_rank": 2,
                "pnp_inlier": False,
                "reprojection_error_px": 40.0,
                "pnp_success": True,
            },
            {
                "scene": "ToyScene",
                "query_id": "query_000002",
                "matched_gaussian_id": "2",
                "matched_landmark_id": "2",
                "descriptor_score": 0.75,
                "match_rank": 1,
                "pnp_inlier": True,
                "reprojection_error_px": 2.0,
                "pnp_success": True,
            },
            {
                "scene": "ToyScene",
                "query_id": "query_000002",
                "matched_gaussian_id": "5",
                "matched_landmark_id": "5",
                "descriptor_score": 0.55,
                "match_rank": 2,
                "pnp_inlier": False,
                "reprojection_error_px": 18.0,
                "pnp_success": True,
            },
            {
                "scene": "ToyScene",
                "query_id": "query_000003",
                "matched_gaussian_id": "6",
                "matched_landmark_id": "6",
                "descriptor_score": 0.8,
                "match_rank": 1,
                "pnp_inlier": True,
                "reprojection_error_px": 1.5,
                "pnp_success": True,
            },
            {
                "scene": "ToyScene",
                "query_id": "query_000004",
                "matched_gaussian_id": "7",
                "matched_landmark_id": "7",
                "descriptor_score": 0.99,
                "match_rank": 1,
                "pnp_inlier": False,
                "reprojection_error_px": 80.0,
                "pnp_success": False,
            },
        ],
        {"scene": "ToyScene", "split_name": split_name},
    )


def _save_v2_bank_with_repeated_keypoint_ids(path):
    records = []
    for image_id, gid in (("seq1/frame00001.png", 1), ("seq1/frame00002.png", 2)):
        records.append(
            {
                "scene": "ToyScene",
                "query_id": f"{image_id}::kp_000000",
                "image_id": image_id,
                "keypoint_id": "kp_000000",
                "matched_gaussian_id": str(gid),
                "matched_landmark_id": str(gid),
                "descriptor_score": 0.95,
                "match_rank": 1,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "dense_transition": "improved",
                "pnp_success": True,
            }
        )
    save_feedback_bank(
        path,
        records,
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )


def test_build_query_conditioned_lsf_artifacts_writes_solver_inputs(tmp_path):
    source_map = tmp_path / "source_map"
    _write_source_map(source_map, [1, 2, 3])
    bank = tmp_path / "feedback_bank.jsonl"
    _save_bank(bank)
    output = tmp_path / "artifacts"

    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--source_map",
            str(source_map),
            "--output_dir",
            str(output),
            "--num_gaussians",
            "8",
            "--hard_query_topk",
            "1",
            "--candidate_pool_size",
            "4",
            "--hard_query_mode",
            "hard_negative_pressure",
            "--positive_reprojection_threshold_px",
            "4.0",
            "--hard_negative_score_threshold",
            "0.65",
            "--hard_negative_reprojection_threshold_px",
            "8.0",
        ]
    )

    assert main(args) == 0

    positive = torch.load(output / "positive_support_binary.pt", map_location="cpu")
    risk = torch.load(output / "hard_negative_risk_binary.pt", map_location="cpu")
    safe_core = torch.load(output / "safe_core_source_positive.pt", map_location="cpu")
    candidates = torch.load(output / "candidate_pool_positive_lowrisk_top4.pt", map_location="cpu")
    episode = torch.load(output / "episode_cache_top1.pt", map_location="cpu")
    summary = json.loads((output / "episode_cache_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert positive.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert risk.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert safe_core.tolist() == [1, 2]
    assert candidates.tolist() == [6]
    assert episode["query_id"].tolist() == [1, 1]
    assert episode["candidate_landmark_ids"].shape == (2, 1)
    assert summary["hard_query_count"] == 1
    assert summary["filtered_record_count"] == 2
    assert manifest["split_name"] == "selfmap_train"
    assert manifest["test_split_used"] is False


def test_build_query_conditioned_lsf_artifacts_rejects_test_split(tmp_path):
    source_map = tmp_path / "source_map"
    _write_source_map(source_map, [1, 2, 3])
    bank = tmp_path / "feedback_bank.jsonl"
    _save_bank(bank, split_name="test")

    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--source_map",
            str(source_map),
            "--output_dir",
            str(tmp_path / "artifacts"),
            "--num_gaussians",
            "8",
        ]
    )

    with pytest.raises(ValueError, match="test split"):
        main(args)


def test_build_query_conditioned_lsf_artifacts_rejects_v2_until_string_ids_are_supported(tmp_path):
    source_map = tmp_path / "source_map"
    _write_source_map(source_map, [1, 2, 3])
    bank = tmp_path / "feedback_bank_v2.jsonl"
    _save_v2_bank_with_repeated_keypoint_ids(bank)
    output = tmp_path / "artifacts"

    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--source_map",
            str(source_map),
            "--output_dir",
            str(output),
            "--num_gaussians",
            "8",
            "--hard_query_topk",
            "2",
            "--positive_reprojection_threshold_px",
            "4.0",
        ]
    )

    with pytest.raises(ValueError, match="does not preserve feedback_bank_v2 string query ids"):
        main(args)
