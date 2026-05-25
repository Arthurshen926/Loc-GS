import json

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import save_feedback_bank


def _valid_record():
    return {
        "scene": "ShopFacade",
        "query_id": "seq1/frame00001.png",
        "image_id": "seq1/frame00001.png",
        "source_view_id": "seq1/frame00001.png",
        "pose_source": "selfmap_rehearsal",
        "keypoint_id": "kp_000123",
        "keypoint_xy": [10.0, 20.0],
        "matched_landmark_id": "101",
        "matched_gaussian_id": "1101",
        "descriptor_score": 0.91,
        "match_rank": 1,
        "pnp_inlier": True,
        "reprojection_error_px": 1.2,
        "depth_consistency": 0.95,
        "visibility_score": 0.8,
        "dense_transition": "rescued",
        "dense_delta_te_cm": -2.0,
    }


def test_feedback_bank_v2_audit_passes_real_query_level_bank(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        [_valid_record()],
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v2(bank_path)

    assert audit["audit_status"] == "passed"
    assert audit["paper_safe_candidate"] is True
    assert audit["record_count"] == 1
    assert audit["query_count"] == 1
    assert audit["image_group_count"] == 1


def test_feedback_bank_v2_audit_reports_image_group_count_separately(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    records = []
    for keypoint_id in ("kp_000000", "kp_000001"):
        record = _valid_record()
        record["query_id"] = f"seq1/frame00001.png::{keypoint_id}"
        record["keypoint_id"] = keypoint_id
        records.append(record)
    save_feedback_bank(
        bank_path,
        records,
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v2(bank_path)

    assert audit["audit_status"] == "passed"
    assert audit["query_count"] == 2
    assert audit["image_group_count"] == 1


def test_feedback_bank_v2_audit_rejects_synthetic_query_ids(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    record = _valid_record()
    record["query_id"] = "query_000000"
    save_feedback_bank(
        bank_path,
        [record],
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "synthetic_index",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v2(bank_path)

    assert audit["audit_status"] == "failed"
    assert audit["paper_safe_candidate"] is False
    assert any("synthetic query_id" in reason for reason in audit["reasons"])


def test_feedback_bank_v2_audit_rejects_synthetic_base_query_and_image_ids(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    record = _valid_record()
    record["query_id"] = "query_000000::kp_000000"
    record["image_id"] = "query_000000"
    save_feedback_bank(
        bank_path,
        [record],
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v2(bank_path)

    assert audit["audit_status"] == "failed"
    assert any("synthetic query_id" in reason for reason in audit["reasons"])
    assert any("synthetic image_id" in reason for reason in audit["reasons"])


def test_feedback_bank_v2_audit_rejects_missing_dense_outcome(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    record = _valid_record()
    record.pop("dense_transition")
    record.pop("dense_delta_te_cm")
    save_feedback_bank(
        bank_path,
        [record],
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    audit = audit_feedback_bank_v2(bank_path)

    assert audit["audit_status"] == "failed"
    assert any("dense outcome" in reason for reason in audit["reasons"])


def test_feedback_bank_v2_audit_payload_is_json_serializable(tmp_path):
    bank_path = tmp_path / "feedback_bank.jsonl"
    save_feedback_bank(
        bank_path,
        [_valid_record()],
        {
            "scene": "ShopFacade",
            "split_name": "selfmap_train",
            "schema_version": "feedback_bank_v2",
            "query_id_source": "image_id",
            "split_audit": {"audit_status": "passed"},
        },
    )

    json.dumps(audit_feedback_bank_v2(bank_path))
