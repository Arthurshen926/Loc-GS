import json
from pathlib import Path

import torch

from loc_gs.scripts.train_internal_landmark_selector import main


def _write_distilled_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 2,
            "split_audit": {
                "schema_version": "internal_split_audit_v1",
                "audit_status": "passed",
                "split_name": "train",
                "official_test_used": False,
                "test_split_used": False,
            },
        },
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[10, 20], [10, 30]], dtype=torch.int64),
        "cosine": torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
        "label": torch.tensor([0, 0], dtype=torch.int64),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "dense_consistent": torch.tensor([[True, False], [False, False]], dtype=torch.bool),
        "sparse_inlier": torch.tensor([[True, False], [True, False]], dtype=torch.bool),
        "reprojection_error": torch.tensor([[1.0, 18.0], [2.0, 22.0]], dtype=torch.float32),
        "solver_weight": torch.tensor([[3.0, 1.5], [2.0, 1.0]], dtype=torch.float32),
        "label_roles": [["protected_support", "hard_negative"], ["positive_inlier", "hard_negative"]],
        "query_id": ["a.png::kp0", "a.png::kp1"],
        "image_id": ["a.png", "a.png"],
        "keypoint_id": ["kp0", "kp1"],
        "source_phase": ["train", "train"],
    }
    torch.save(payload, path)
    return path


def test_train_internal_landmark_selector_cli_writes_bundle(tmp_path: Path):
    out = tmp_path / "selector"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(_write_distilled_artifact(tmp_path / "distilled.pt")),
            "--output_dir",
            str(out),
            "--conflict_penalty",
            "0.25",
        ]
    )

    assert rc == 0
    selector = json.loads((out / "landmark_selector.json").read_text(encoding="utf-8"))
    conflicts = json.loads((out / "conflict_graph.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))

    assert selector["schema_version"] == "internal_landmark_selector_v1"
    assert conflicts["schema_version"] == "internal_conflict_graph_v1"
    assert float(selector["landmark_scores"]["10"]) > float(selector["landmark_scores"]["20"])
    assert summary["student_modules"] == ["landmark_selector", "conflict_graph"]
    assert summary["protected_support_count"] == 1
    assert summary["positive_inlier_count"] == 1
    assert manifest["inference_stage"] == "sparse_landmark_selector_training"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert manifest["hyperparameters"]["conflict_penalty"] == 0.25
    assert split_audit["audit_status"] == "passed"
