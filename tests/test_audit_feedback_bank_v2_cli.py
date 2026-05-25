import json
import subprocess
import sys

from loc_gs.feedback.io import save_feedback_bank


def test_audit_feedback_bank_v2_cli_writes_audit_json(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        [
            {
                "scene": "ShopFacade",
                "query_id": "seq1/frame00001.png",
                "image_id": "seq1/frame00001.png",
                "source_view_id": "seq1/frame00001.png",
                "pose_source": "selfmap_rehearsal",
                "keypoint_id": "kp0",
                "keypoint_xy": [1.0, 2.0],
                "matched_landmark_id": "7",
                "matched_gaussian_id": "70",
                "descriptor_score": 0.9,
                "pnp_inlier": True,
                "reprojection_error_px": 0.5,
                "dense_transition": "stable_ok",
            }
        ],
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.audit_feedback_bank_v2",
            "--feedback_bank",
            str(bank_path),
            "--output_json",
            str(output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["audit_status"] == "passed"
    assert payload["paper_safe_candidate"] is True
