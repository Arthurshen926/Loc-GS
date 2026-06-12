import json
from pathlib import Path

import torch

from loc_gs.scripts.train_internal_descriptor_fusion import main


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
        "query_desc": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "landmark_desc": torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[10, 20], [10, 30]], dtype=torch.int64),
        "cosine": torch.tensor([[0.1, 0.9], [0.8, 0.7]], dtype=torch.float32),
        "label": torch.tensor([0, 1], dtype=torch.int64),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "solver_weight": torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32),
        "label_roles": [["protected_support", "hard_negative"], ["hard_negative", "protected_support"]],
        "query_id": ["a.png::kp0", "a.png::kp1"],
        "image_id": ["a.png", "a.png"],
        "keypoint_id": ["kp0", "kp1"],
        "source_phase": ["train", "train"],
    }
    torch.save(payload, path)
    return path


def test_train_internal_descriptor_fusion_cli_writes_bundle(tmp_path: Path):
    out = tmp_path / "fusion"

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
            "--trust_region",
            "0.75",
            "--score_scale",
            "0.01",
        ]
    )

    assert rc == 0
    fusion = json.loads((out / "descriptor_fusion.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))

    assert fusion["schema_version"] == "internal_descriptor_fusion_v1"
    assert (out / "descriptor_fusion.pt").is_file()
    assert fusion["score_scale"] == 0.01
    assert fusion["fused_descriptors"]["10"][0] > fusion["fused_descriptors"]["10"][1]
    assert summary["student_modules"] == ["descriptor_fusion"]
    assert summary["protected_support_count"] == 2
    assert summary["hard_negative_count"] == 2
    assert manifest["inference_stage"] == "sparse_descriptor_fusion_training"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert manifest["hyperparameters"]["trust_region"] == 0.75
    assert split_audit["audit_status"] == "passed"
