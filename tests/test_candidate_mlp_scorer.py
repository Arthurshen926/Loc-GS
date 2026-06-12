import json
from pathlib import Path

import torch

from loc_gs.scripts.train_internal_candidate_mlp_scorer import main
from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.students.candidate_mlp_scorer import (
    CandidateMLPScorerConfig,
    candidate_mlp_score_rows,
    load_candidate_mlp_scorer,
    train_candidate_mlp_scorer,
)


def _descriptor_batch() -> SparseCandidateBatch:
    return SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        query_descriptors=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        candidate_landmark_ids=[[1, 2], [3, 4], [5, 6], [7, 8]],
        candidate_scores=[[0.95, 0.10], [0.90, 0.20], [0.85, 0.30], [0.80, 0.40]],
        candidate_landmark_descriptors=[
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        candidate_valid_mask=[[True, True], [True, True], [True, True], [True, True]],
        candidate_geometric_correct=[[False, True], [False, True], [False, True], [False, True]],
        candidate_dense_consistent=[[False, True], [False, True], [False, True], [False, True]],
        candidate_sparse_inlier=[[False, True], [False, True], [False, True], [False, True]],
        candidate_reprojection_error_px=[[18.0, 1.0], [16.0, 1.5], [20.0, 0.5], [22.0, 1.0]],
        candidate_solver_weight=[[0.25, 3.0], [0.25, 3.0], [0.25, 3.0], [0.25, 3.0]],
        candidate_label_roles=[
            ["hard_negative", "protected_support"],
            ["hard_negative", "protected_support"],
            ["hard_negative", "protected_support"],
            ["hard_negative", "protected_support"],
        ],
    )


def _artifact() -> CachedCandidateArtifact:
    return CachedCandidateArtifact(
        scene="GreatCourt",
        split_name="train",
        source_path="memory",
        artifact_format="listwise",
        topk=2,
        batches=[_descriptor_batch()],
        metadata={"source_split_name": "train"},
    )


def _write_artifact(path: Path) -> Path:
    batch = _descriptor_batch()
    payload = {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_desc": torch.tensor(batch.query_descriptors, dtype=torch.float32),
        "landmark_desc": torch.tensor(batch.candidate_landmark_descriptors, dtype=torch.float32),
        "query_yx": torch.tensor([[xy[1], xy[0]] for xy in batch.keypoint_xy], dtype=torch.float32),
        "landmark_id": torch.tensor(batch.candidate_landmark_ids, dtype=torch.int64),
        "cosine": torch.tensor(batch.candidate_scores, dtype=torch.float32),
        "label": torch.ones(4, dtype=torch.int64),
        "candidate_mask": torch.ones((4, 2), dtype=torch.bool),
        "dense_consistent": torch.tensor(batch.candidate_dense_consistent, dtype=torch.bool),
        "sparse_inlier": torch.tensor(batch.candidate_sparse_inlier, dtype=torch.bool),
        "reprojection_error": torch.tensor(batch.candidate_reprojection_error_px, dtype=torch.float32),
        "solver_weight": torch.tensor(batch.candidate_solver_weight, dtype=torch.float32),
        "label_roles": batch.candidate_label_roles,
        "query_id": [f"img.png::kp{i}" for i in range(4)],
        "image_id": ["img.png"] * 4,
        "keypoint_id": [f"kp{i}" for i in range(4)],
        "source_phase": ["train"] * 4,
    }
    torch.save(payload, path)
    return path


def test_candidate_mlp_scorer_learns_descriptor_pair_reranking(tmp_path: Path):
    model, summary = train_candidate_mlp_scorer(
        _artifact(),
        CandidateMLPScorerConfig(
            epochs=160,
            learning_rate=0.03,
            hidden_dim=8,
            seed=7,
            listwise_loss_weight=1.0,
            batch_size=2,
        ),
    )

    rows = candidate_mlp_score_rows(_descriptor_batch(), model)
    assert summary["student_modules"] == ["candidate_mlp_scorer"]
    assert summary["feature_input_policy"] == "inference_safe"
    assert summary["paper_safe_sparse_inference"] is True
    assert summary["native_top1_correct"] == 0
    assert summary["trained_top1_correct"] == 4
    assert summary["listwise_loss_weight"] == 1.0
    assert summary["batch_size"] == 2
    assert summary["optimizer_step_count"] > summary["epochs"]
    assert summary["score_calibration"] == "train_logit_zscore"
    assert all(row[1] > row[0] for row in rows)

    path = tmp_path / "model.pt"
    torch.save(model.to_torch_dict(), path)
    payload = torch.load(path, map_location="cpu")
    assert payload["score_calibration"] == "train_logit_zscore"
    assert float(payload["logit_std"]) > 0.0
    loaded = load_candidate_mlp_scorer(path)
    assert candidate_mlp_score_rows(_descriptor_batch(), loaded) == rows


def test_train_internal_candidate_mlp_scorer_cli_writes_pt_bundle(tmp_path: Path):
    out = tmp_path / "mlp"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(_write_artifact(tmp_path / "pairs.pt")),
            "--output_dir",
            str(out),
            "--epochs",
            "60",
            "--learning_rate",
            "0.03",
            "--hidden_dim",
            "8",
            "--listwise_loss_weight",
            "1.0",
            "--batch_size",
            "2",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    payload = torch.load(out / "model.pt", map_location="cpu")
    assert payload["schema_version"] == "internal_candidate_mlp_scorer_v1"
    assert payload["score_calibration"] == "train_logit_zscore"
    assert summary["student_modules"] == ["candidate_mlp_scorer"]
    assert summary["listwise_loss_weight"] == 1.0
    assert summary["batch_size"] == 2
    assert summary["optimizer_step_count"] > summary["epochs"]
    assert summary["score_calibration"] == "train_logit_zscore"
    assert summary["feature_input_policy"] == "inference_safe"
    assert manifest["inference_stage"] == "sparse_candidate_mlp_scorer_training"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert manifest["hyperparameters"]["batch_size"] == 2
