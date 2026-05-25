import json
import pickle
import subprocess
import sys

import torch

from loc_gs.feedback.io import save_feedback_bank


def test_build_query_conditioned_solver_constraints_cli_writes_exporter_payload(tmp_path):
    cache = tmp_path / "episode.pt"
    source_idx = tmp_path / "sampled_idx.pkl"
    output = tmp_path / "constraints.json"
    torch.save(
        {
            "query_id": torch.tensor([7, 7], dtype=torch.long),
            "candidate_landmark_ids": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            "candidate_cosine": torch.tensor([[0.9, 0.8], [0.7, 0.4]], dtype=torch.float32),
            "candidate_reprojection_error": torch.tensor([[1.0, 2.0], [1.0, 1.0]], dtype=torch.float32),
            "candidate_visible": torch.ones((2, 2), dtype=torch.bool),
            "candidate_pnp_inlier": torch.ones((2, 2), dtype=torch.bool),
            "metadata": {"split_name": "selfmap_train_rendered"},
        },
        cache,
    )
    with source_idx.open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_query_conditioned_solver_constraints",
            "--episode_cache",
            str(cache),
            "--source_idx",
            str(source_idx),
            "--output_json",
            str(output),
            "--num_gaussians",
            "4",
            "--hard_query_ids",
            "7",
            "--reprojection_threshold_px",
            "3.0",
            "--score_threshold",
            "0.5",
            "--min_candidate_positive",
            "0.5",
            "--min_logdet_delta",
            "0.0",
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "loc_gs_solver_admissibility_v1"
    assert payload["hard_query_ids"] == [7]
    assert payload["candidate_gain"]["2"]["7"]["support"] == 1.0
    assert "3" not in payload["candidate_gain"]
    assert payload["source_loss"]["0"]["7"]["support"] == 1.0
    assert payload["thresholds"]["min_logdet_delta"] == 0.0


def test_build_feedback_bank_v2_solver_admissibility_cli_writes_v3_payload(tmp_path):
    bank = tmp_path / "feedback_bank.jsonl"
    source_idx = tmp_path / "sampled_idx.pkl"
    output = tmp_path / "constraints_v3.json"
    save_feedback_bank(
        bank,
        [
            {
                "scene": "ToyScene",
                "query_id": "seq1/frame00001.png::kp_000000",
                "image_id": "seq1/frame00001.png",
                "source_view_id": "seq1/frame00001.png",
                "pose_source": "selfmap_rehearsal",
                "keypoint_id": "kp_000000",
                "keypoint_xy": [10.0, 20.0],
                "matched_landmark_id": "0",
                "matched_gaussian_id": "0",
                "descriptor_score": 0.9,
                "match_rank": 1,
                "pnp_inlier": True,
                "pnp_success": True,
                "reprojection_error_px": 1.0,
                "visibility_score": 1.0,
                "dense_refine_success": True,
                "dense_transition": "improved",
                "dense_delta_te_cm": -1.0,
            },
            {
                "scene": "ToyScene",
                "query_id": "seq1/frame00001.png::kp_000001",
                "image_id": "seq1/frame00001.png",
                "source_view_id": "seq1/frame00001.png",
                "pose_source": "selfmap_rehearsal",
                "keypoint_id": "kp_000001",
                "keypoint_xy": [12.0, 22.0],
                "matched_landmark_id": "2",
                "matched_gaussian_id": "2",
                "descriptor_score": 0.8,
                "match_rank": 1,
                "pnp_inlier": True,
                "pnp_success": True,
                "reprojection_error_px": 2.0,
                "visibility_score": 1.0,
                "dense_refine_success": False,
                "dense_transition": "worsened",
                "dense_delta_te_cm": 6.0,
            },
        ],
        {
            "scene": "ToyScene",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )
    with source_idx.open("wb") as handle:
        pickle.dump(torch.tensor([0, 1], dtype=torch.long), handle)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_feedback_bank_v2_solver_admissibility",
            "--feedback_bank",
            str(bank),
            "--source_idx",
            str(source_idx),
            "--output_json",
            str(output),
            "--num_gaussians",
            "4",
            "--hard_query_topk",
            "1",
            "--cvar_alpha",
            "0.5",
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "loc_gs_solver_admissibility_v2_feedback_bank"
    assert payload["hard_query_ids"] == ["seq1/frame00001.png"]
    assert payload["candidate_gain"]["2"]["seq1/frame00001.png"]["dense_worsen_risk"] == 1.0
    assert payload["thresholds"]["cvar_alpha"] == 0.5
