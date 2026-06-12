import json
from pathlib import Path

import torch

from loc_gs.scripts.run_internal_sparse_gate import main


def _write_pair_cache(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "feedback_bank_split_name": "selfmap_train_rendered",
            "topk": 2,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[11, 12], [21, 22]], dtype=torch.int64),
        "cosine": torch.tensor([[0.5, 0.4], [0.8, 0.7]], dtype=torch.float32),
        "label": torch.tensor([0, 2], dtype=torch.int64),
        "candidate_mask": torch.tensor([[True, True], [True, True]], dtype=torch.bool),
        "query_id": ["img.png::kp0", "img.png::kp1"],
        "image_id": ["img.png", "img.png"],
        "keypoint_id": ["kp0", "kp1"],
        "source_phase": ["train", "train"],
    }
    torch.save(payload, path)
    return path


def _write_metrics(path: Path, *, median_te_cm: float, split_name: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "scene": "GreatCourt",
                "split_name": split_name,
                "median_te_cm": median_te_cm,
                "median_re_deg": 0.1,
                "query_count": 7,
                "recall_10cm_5d": 0.25,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_internal_sparse_gate_cli_writes_manifest_metrics_and_candidate_preview(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _write_metrics(tmp_path / "candidate.json", median_te_cm=13.0, split_name="train_dev_seed13_20p")
    out = tmp_path / "gate"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev_seed13_20p",
            "--candidate_artifact",
            str(pair_cache),
            "--baseline_metrics",
            str(baseline),
            "--candidate_metrics",
            str(candidate),
            "--output_dir",
            str(out),
            "--dense_target_cm",
            "10.0",
            "--max_export_batches",
            "1",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    preview = (out / "candidate_batches_preview.jsonl").read_text(encoding="utf-8")

    assert manifest["scene"] == "GreatCourt"
    assert manifest["split_name"] == "train_dev_seed13_20p"
    assert manifest["inference_stage"] == "sparse_only"
    assert manifest["dense_teacher_enabled"] is False
    assert manifest["hyperparameters"]["dense_target_cm"] == 10.0
    assert manifest["hyperparameters"]["max_export_batches"] == 1
    assert metrics["sparse_gate_status"] == "improved_not_target"
    assert metrics["baseline_median_te_cm"] == 15.0
    assert metrics["candidate_median_te_cm"] == 13.0
    assert metrics["candidate_artifact"]["topk_available"] == 1
    assert '"query_id": "img.png"' in preview
    assert "loc_gs.scripts.run_internal_sparse_gate" in (out / "command.txt").read_text(encoding="utf-8")
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert split_audit["audit_status"] == "passed"
    assert (out / "git_status.txt").exists()


def test_internal_sparse_gate_marks_unverified_internal_pipeline_metrics_diagnostic(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train")
    candidate = _write_metrics(tmp_path / "candidate.json", median_te_cm=500.0, split_name="train")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    data["schema_version"] = "internal_sparse_smoke_metrics_v1"
    data["pose_metric_status"] = "computed_unverified"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "gate"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(pair_cache),
            "--baseline_metrics",
            str(baseline),
            "--candidate_metrics",
            str(candidate),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["sparse_gate_status"] == "diagnostic_pose_frame_unverified"
    assert metrics["candidate_metric_source"] == "internal_sparse_smoke_metrics_v1"
    assert metrics["candidate_pose_metric_status"] == "computed_unverified"
