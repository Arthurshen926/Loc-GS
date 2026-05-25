import json
import subprocess
import sys

import torch

from loc_gs.dense_support.distill_dense_lsf import distill_dense_lsf_targets
from loc_gs.feedback.io import save_feedback_bank


def _write_dense_feedback_bank(path):
    records = [
        {
            "scene": "ToyScene",
            "query_id": "img0::kp0",
            "image_id": "img0",
            "keypoint_id": "kp0",
            "matched_landmark_id": "10",
            "matched_gaussian_id": "10",
            "descriptor_score": 0.9,
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "dense_transition": "improved",
            "dense_delta_te_cm": -4.0,
        },
        {
            "scene": "ToyScene",
            "query_id": "img1::kp0",
            "image_id": "img1",
            "keypoint_id": "kp0",
            "matched_landmark_id": "11",
            "matched_gaussian_id": "11",
            "descriptor_score": 0.8,
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "dense_transition": "worsened",
            "dense_delta_te_cm": 6.0,
        },
        {
            "scene": "ToyScene",
            "query_id": "img2::kp0",
            "image_id": "img2",
            "keypoint_id": "kp0",
            "matched_landmark_id": "12",
            "matched_gaussian_id": "12",
            "descriptor_score": 0.7,
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "dense_transition": "unchanged",
            "dense_delta_te_cm": 0.0,
        },
    ]
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


def test_distill_dense_lsf_targets_suppresses_dense_worsened_gaussians(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_dense_feedback_bank(bank_path)

    payload = distill_dense_lsf_targets(bank_path, num_gaussians=16, smoothing=0.5)

    target = payload["dense_lsf_target"]
    assert target[10] > 0.5
    assert target[11] < 0.5
    assert target[12] == 0.5
    assert payload["observed_count"][10] == 1
    assert payload["selected_idx"].tolist() == [10, 12]
    assert payload["metadata"]["feedback_bank_schema"] == "feedback_bank_v2"
    assert payload["metadata"]["usage_scope"] == "dense_residual_teacher_only"
    assert payload["metadata"]["sparse_selector_safe"] is False


def test_distill_dense_lsf_cli_writes_targets_and_summary(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    _write_dense_feedback_bank(bank_path)
    output_pt = tmp_path / "dense_lsf_targets.pt"
    output_json = tmp_path / "dense_lsf_summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.distill_dense_lsf_from_feedback",
            "--feedback_bank",
            str(bank_path),
            "--num_gaussians",
            "16",
            "--output_pt",
            str(output_pt),
            "--output_json",
            str(output_json),
        ],
        check=True,
    )

    payload = torch.load(output_pt, map_location="cpu")
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["dense_lsf_target"][10] > payload["dense_lsf_target"][11]
    assert summary["observed_gaussian_count"] == 3
    assert summary["selected_count"] == 2
    assert summary["usage_scope"] == "dense_residual_teacher_only"
    assert summary["sparse_selector_safe"] is False
