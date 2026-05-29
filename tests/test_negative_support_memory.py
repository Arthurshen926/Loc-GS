import json
import subprocess
import sys

from loc_gs.feedback.io import save_feedback_bank
from loc_gs.stdloc_native.negative_support_memory import (
    build_negative_support_graph,
    conflict_delta,
    conflict_penalty,
)


def test_negative_support_memory_adds_pairwise_conflict_for_competing_hard_negatives():
    graph = build_negative_support_graph(
        [
            {"query_id": "q0", "keypoint_id": "kp0", "landmark_id": 10, "pnp_inlier": True, "score": 0.9},
            {"query_id": "q0", "keypoint_id": "kp0", "landmark_id": 20, "pnp_inlier": False, "score": 0.8},
            {"query_id": "q0", "keypoint_id": "kp0", "landmark_id": 30, "pnp_inlier": False, "score": 0.7},
            {"query_id": "q1", "keypoint_id": "kp1", "landmark_id": 20, "pnp_inlier": False, "score": 0.5},
        ],
        min_score=0.1,
    )

    assert graph["edge_weights"][(20, 30)] > 0.0
    assert graph["unary_risk"][20] > graph["unary_risk"].get(10, 0.0)
    assert conflict_penalty([20, 30], graph) > conflict_penalty([20], graph)
    assert conflict_delta(add_id=30, drop_id=10, selected_ids=[10, 20], graph=graph, pair_weight=1.0) > 0.0


def _write_feedback_bank(path, *, split_name="selfmap_train"):
    records = [
        {
            "scene": "ToyScene",
            "query_id": "img0::kp0",
            "image_id": "img0",
            "keypoint_id": "kp0",
            "matched_landmark_id": "20",
            "matched_gaussian_id": "20",
            "descriptor_score": 0.8,
            "pnp_inlier": False,
            "reprojection_error_px": 10.0,
            "dense_transition": "worsened",
            "dense_delta_te_cm": 3.0,
        },
        {
            "scene": "ToyScene",
            "query_id": "img0::kp0",
            "image_id": "img0",
            "keypoint_id": "kp0",
            "matched_landmark_id": "30",
            "matched_gaussian_id": "30",
            "descriptor_score": 0.7,
            "pnp_inlier": False,
            "reprojection_error_px": 11.0,
            "dense_transition": "worsened",
            "dense_delta_te_cm": 4.0,
        },
    ]
    save_feedback_bank(
        path,
        records,
        {
            "scene": "ToyScene",
            "split_name": split_name,
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )


def test_build_negative_support_graph_cli_writes_audited_artifact(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    output_pt = tmp_path / "negative_support_graph.pt"
    output_json = tmp_path / "summary.json"
    _write_feedback_bank(bank_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_negative_support_graph",
            "--feedback_bank",
            str(bank_path),
            "--output_pt",
            str(output_pt),
            "--output_json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    split_audit = json.loads((output_pt.parent / "split_audit.json").read_text(encoding="utf-8"))
    assert summary["edge_count"] == 1
    assert summary["feedback_bank_audit_status"] == "passed"
    assert split_audit["audit_status"] == "passed"


def test_build_negative_support_graph_cli_rejects_test_split(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    output_pt = tmp_path / "negative_support_graph.pt"
    output_json = tmp_path / "summary.json"
    _write_feedback_bank(bank_path, split_name="test")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_negative_support_graph",
            "--feedback_bank",
            str(bank_path),
            "--output_pt",
            str(output_pt),
            "--output_json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split feedback banks are not allowed" in result.stderr
