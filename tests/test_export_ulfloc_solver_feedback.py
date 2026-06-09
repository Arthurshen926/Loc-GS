import json
import pickle

import pytest
import torch

from loc_gs.scripts.export_ulfloc_solver_feedback import (
    build_argparser,
    build_ulfloc_solver_feedback,
    main,
)


def _write_support_artifact(path, *, split_name="selfmap_train"):
    torch.save(
        {
            "support_score": torch.tensor([0.0, 0.8, 0.4, 0.2], dtype=torch.float32),
            "hard_negative_risk": torch.tensor([0.0, 0.0, 0.5, 0.0], dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([0.0, 0.0, 0.0, 0.7], dtype=torch.float32),
            "observed_count": torch.tensor([0, 5, 3, 4], dtype=torch.long),
            "metadata": {
                "scene": "ShopFacade",
                "split": split_name,
                "source_feedback_bank": "feedback_bank.jsonl",
            },
        },
        path,
    )


def test_export_ulfloc_solver_feedback_writes_landmark_weights_and_audit_files(tmp_path):
    support = tmp_path / "solver_support.pt"
    _write_support_artifact(support)
    output_dir = tmp_path / "ulf_feedback"

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--solver_support",
            str(support),
            "--output_dir",
            str(output_dir),
            "--alpha",
            "0.5",
            "--risk_penalty",
            "1.0",
            "--min_weight",
            "0.25",
            "--max_weight",
            "1.75",
        ]
    )

    assert main(args) == 0

    artifact_path = output_dir / "solver_feedback.pkl"
    with artifact_path.open("rb") as f:
        artifact = pickle.load(f)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    assert artifact["schema_version"] == "ulfloc_solver_feedback_v1"
    assert artifact["split_name"] == "selfmap_train"
    assert weights.shape == (4,)
    assert weights[1] > 1.0
    assert weights[2] < weights[1]
    assert weights[3] < 1.0
    assert manifest["scene"] == "ShopFacade"
    assert manifest["test_split_used"] is False
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "split_audit.json").exists()
    assert (output_dir / "git_status.txt").exists()


def test_export_ulfloc_solver_feedback_rejects_test_split(tmp_path):
    support = tmp_path / "solver_support.pt"
    _write_support_artifact(support, split_name="test")

    args = build_argparser().parse_args(
        [
            "--scene",
            "ShopFacade",
            "--solver_support",
            str(support),
            "--output_dir",
            str(tmp_path / "ulf_feedback"),
        ]
    )

    with pytest.raises(ValueError, match="test split"):
        main(args)


def test_build_ulfloc_solver_feedback_uses_artifact_risk_as_solver_risk():
    artifact = build_ulfloc_solver_feedback(
        {
            "support_score": torch.tensor([0.8, 0.8], dtype=torch.float32),
            "hard_negative_risk": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "dense_worsen_risk": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "artifact_risk": torch.tensor([0.0, 0.9], dtype=torch.float32),
            "observed_count": torch.tensor([4, 4], dtype=torch.long),
            "metadata": {"split_name": "selfmap_train"},
        },
        scene="ShopFacade",
        alpha=0.5,
        risk_penalty=1.0,
        min_weight=0.25,
        max_weight=1.75,
    )

    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    risk = torch.as_tensor(artifact["landmark_risk"], dtype=torch.float32)
    assert risk.tolist() == pytest.approx([0.0, 0.9])
    assert weights[0] > 1.0
    assert weights[1] < 1.0


def test_guarded_solver_feedback_keeps_weak_evidence_native_and_allows_clear_gain_or_risk():
    artifact = build_ulfloc_solver_feedback(
        {
            "support_score": torch.tensor([0.9, 0.55, 0.1, 0.8], dtype=torch.float32),
            "hard_negative_risk": torch.tensor([0.0, 0.0, 0.9, 0.8], dtype=torch.float32),
            "dense_worsen_risk": torch.zeros(4, dtype=torch.float32),
            "observed_count": torch.tensor([6, 6, 6, 6], dtype=torch.long),
            "positive_observed_count": torch.tensor([4, 1, 0, 5], dtype=torch.long),
            "metadata": {"split_name": "selfmap_train"},
        },
        scene="ShopFacade",
        alpha=0.5,
        risk_penalty=1.0,
        min_weight=0.25,
        max_weight=1.75,
        guarded=True,
        min_boost_support=0.7,
        min_boost_positive_count=3,
        max_boost_risk=0.2,
        min_deboost_risk=0.6,
        max_deboost_positive_count=1,
    )

    weights = torch.as_tensor(artifact["landmark_weights"], dtype=torch.float32)
    assert weights[0] > 1.0
    assert weights[1].item() == pytest.approx(1.0)
    assert weights[2] < 1.0
    assert weights[3].item() == pytest.approx(1.0)
    assert artifact["metadata"]["guarded_summary"]["boosted_count"] == 1
    assert artifact["metadata"]["guarded_summary"]["deboosted_count"] == 1


def test_build_ulfloc_solver_feedback_exports_view_dependent_sparse_support():
    artifact = build_ulfloc_solver_feedback(
        {
            "support_score": torch.tensor([0.0, 0.8, 0.4, 0.2], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(4, dtype=torch.float32),
            "observed_count": torch.tensor([0, 5, 3, 4], dtype=torch.long),
            "metadata": {"split_name": "selfmap_train"},
            "per_query_support": {
                "view_a": {1: 4.0, 3: 1.0},
                "view_b": {"2": 2.0},
            },
        },
        scene="ShopFacade",
        alpha=0.5,
        risk_penalty=1.0,
        min_weight=0.25,
        max_weight=1.75,
    )

    view_weights = artifact["view_landmark_weights"]

    assert set(view_weights) == {"view_a", "view_b"}
    assert torch.as_tensor(view_weights["view_a"]["landmark_ids"]).tolist() == [1, 3]
    assert torch.as_tensor(view_weights["view_a"]["weights"])[0] > torch.as_tensor(
        view_weights["view_a"]["weights"]
    )[1]
    assert torch.as_tensor(view_weights["view_b"]["landmark_ids"]).tolist() == [2]
    assert artifact["metadata"]["view_landmark_weight_summary"]["view_count"] == 2
    assert artifact["metadata"]["view_landmark_weight_summary"]["entry_count"] == 3


def test_build_ulfloc_solver_feedback_exports_view_dependent_negative_deboost():
    artifact = build_ulfloc_solver_feedback(
        {
            "support_score": torch.tensor([0.0, 0.8, 0.4, 0.2], dtype=torch.float32),
            "hard_negative_risk": torch.zeros(4, dtype=torch.float32),
            "observed_count": torch.tensor([0, 5, 3, 4], dtype=torch.long),
            "metadata": {"split_name": "selfmap_train"},
            "per_query_support": {
                "view_a": {1: 4.0},
            },
            "per_query_negative_support": {
                "view_a": {1: 1.0, 2: 3.0},
                "view_b": {3: 2.0},
            },
        },
        scene="ShopFacade",
        alpha=0.5,
        risk_penalty=1.0,
        min_weight=0.25,
        max_weight=1.75,
    )

    view_weights = artifact["view_landmark_weights"]
    view_a_ids = torch.as_tensor(view_weights["view_a"]["landmark_ids"]).tolist()
    view_a_weights = torch.as_tensor(view_weights["view_a"]["weights"], dtype=torch.float32)
    by_id = dict(zip(view_a_ids, view_a_weights.tolist()))

    assert by_id[1] > 1.0
    assert by_id[2] < 1.0
    assert torch.as_tensor(view_weights["view_b"]["landmark_ids"]).tolist() == [3]
    assert torch.as_tensor(view_weights["view_b"]["weights"], dtype=torch.float32)[0] < 1.0
    assert artifact["metadata"]["view_landmark_weight_summary"]["negative_entry_count"] == 3
