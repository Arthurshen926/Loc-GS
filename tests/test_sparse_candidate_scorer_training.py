import json
from pathlib import Path

import torch

from loc_gs.scripts.train_internal_sparse_candidate_scorer import main
from loc_gs.sparse.artifact_adapter import CachedCandidateArtifact, load_listwise_candidate_artifact
from loc_gs.sparse.correspondences import SparseCandidateBatch
from loc_gs.training.sparse_candidate_scorer import (
    CandidateScorerConfig,
    score_candidate_rows,
    train_candidate_scorer,
)


def _write_artifact(path: Path) -> Path:
    payload = {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64),
        "cosine": torch.tensor([[0.9, 0.2], [0.8, 0.1], [0.7, 0.05]], dtype=torch.float32),
        "label": torch.tensor([1, 1, 1], dtype=torch.int64),
        "candidate_mask": torch.ones((3, 2), dtype=torch.bool),
        "query_id": ["img.png::kp0", "img.png::kp1", "img.png::kp2"],
        "image_id": ["img.png", "img.png", "img.png"],
        "keypoint_id": ["kp0", "kp1", "kp2"],
        "source_phase": ["train", "train", "train"],
    }
    torch.save(payload, path)
    return path


def _write_distilled_artifact(path: Path) -> Path:
    payload = {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        "cosine": torch.tensor([[0.95, 0.10], [0.90, 0.20]], dtype=torch.float32),
        "label": torch.tensor([1, 1], dtype=torch.int64),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "dense_consistent": torch.tensor([[False, True], [False, True]], dtype=torch.bool),
        "sparse_inlier": torch.tensor([[False, True], [False, True]], dtype=torch.bool),
        "reprojection_error": torch.tensor([[18.0, 1.0], [16.0, 1.5]], dtype=torch.float32),
        "solver_weight": torch.tensor([[0.25, 3.0], [0.25, 3.0]], dtype=torch.float32),
        "label_roles": [["hard_negative", "protected_support"], ["hard_negative", "protected_support"]],
        "query_id": ["img.png::kp0", "img.png::kp1"],
        "image_id": ["img.png", "img.png"],
        "keypoint_id": ["kp0", "kp1"],
        "source_phase": ["train", "train"],
    }
    torch.save(payload, path)
    return path


def test_candidate_scorer_training_learns_rank_bias_from_labels(tmp_path: Path):
    artifact = load_listwise_candidate_artifact(_write_artifact(tmp_path / "pairs.pt"))

    model, summary = train_candidate_scorer(
        artifact,
        CandidateScorerConfig(epochs=80, learning_rate=0.5, rank_feature_scale=1.0),
    )

    scored = score_candidate_rows(artifact.batches[0], model)
    assert summary["label_count"] == 3
    assert summary["trained_top1_correct"] == 3
    assert all(row[0]["geometric_correct"] is True for row in scored)


def test_candidate_scorer_training_uses_dense_teacher_consistency():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        candidate_landmark_ids=[[1, 2], [3, 4], [5, 6]],
        candidate_scores=[[0.95, 0.10], [0.90, 0.20], [0.85, 0.30]],
        candidate_valid_mask=[[True, True], [True, True], [True, True]],
        candidate_geometric_correct=[[True, True], [True, True], [True, True]],
        candidate_dense_consistent=[[False, True], [False, True], [False, True]],
        candidate_sparse_inlier=[[False, True], [False, True], [False, True]],
        candidate_reprojection_error_px=[[18.0, 1.0], [16.0, 1.5], [14.0, 2.0]],
        candidate_solver_weight=[[0.25, 3.0], [0.25, 3.0], [0.25, 3.0]],
    )
    artifact = CachedCandidateArtifact(
        scene="GreatCourt",
        split_name="train",
        source_path="memory",
        artifact_format="listwise",
        topk=2,
        batches=[batch],
        metadata={},
    )

    model, summary = train_candidate_scorer(
        artifact,
        CandidateScorerConfig(epochs=100, learning_rate=0.5),
    )

    scored = score_candidate_rows(batch, model)
    assert summary["dense_teacher_sample_count"] == 6
    assert summary["weighted_sample_count"] > summary["sample_count"]
    assert model.feature_names == ("native_score", "negative_rank", "valid")
    assert all(row[0]["candidate_rank"] == 1 for row in scored)


def test_candidate_scorer_training_uses_inference_safe_match_context_features():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        candidate_landmark_ids=[[1, 2], [3, 4], [5, 6]],
        candidate_scores=[[0.95, 0.10], [0.90, 0.20], [0.85, 0.30]],
        candidate_valid_mask=[[True, True], [True, True], [True, True]],
        candidate_geometric_correct=[[False, True], [False, True], [False, True]],
        candidate_landmark_prior=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        candidate_margin=[[0.85, 0.85], [0.70, 0.70], [0.55, 0.55]],
        candidate_query_score=[[0.9, 0.9], [0.8, 0.8], [0.7, 0.7]],
    )
    artifact = CachedCandidateArtifact(
        scene="GreatCourt",
        split_name="train",
        source_path="memory",
        artifact_format="listwise",
        topk=2,
        batches=[batch],
        metadata={},
    )

    model, summary = train_candidate_scorer(
        artifact,
        CandidateScorerConfig(
            epochs=120,
            learning_rate=0.5,
            feature_names=("native_score", "negative_rank", "valid", "landmark_prior", "margin", "query_score"),
        ),
    )

    scored = score_candidate_rows(batch, model)
    assert summary["feature_input_policy"] == "inference_safe"
    assert summary["paper_safe_sparse_inference"] is True
    assert all(row[0]["candidate_rank"] == 1 for row in scored)


def test_candidate_scorer_training_uses_inference_safe_descriptor_pair_features():
    batch = SparseCandidateBatch(
        scene="GreatCourt",
        split_name="train",
        query_id="img.png",
        keypoint_xy=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        query_descriptors=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        candidate_landmark_ids=[[1, 2], [3, 4], [5, 6]],
        candidate_scores=[[0.95, 0.10], [0.90, 0.20], [0.85, 0.30]],
        candidate_landmark_descriptors=[
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        candidate_valid_mask=[[True, True], [True, True], [True, True]],
        candidate_geometric_correct=[[False, True], [False, True], [False, True]],
    )
    artifact = CachedCandidateArtifact(
        scene="GreatCourt",
        split_name="train",
        source_path="memory",
        artifact_format="listwise",
        topk=2,
        batches=[batch],
        metadata={},
    )

    model, summary = train_candidate_scorer(
        artifact,
        CandidateScorerConfig(
            epochs=120,
            learning_rate=0.5,
            feature_names=(
                "native_score",
                "negative_rank",
                "valid",
                "descriptor_alignment",
                "negative_descriptor_l2",
                "negative_descriptor_abs_diff_mean",
            ),
        ),
    )

    scored = score_candidate_rows(batch, model)
    assert summary["feature_input_policy"] == "inference_safe"
    assert summary["paper_safe_sparse_inference"] is True
    assert summary["teacher_only_feature_names"] == []
    assert all(row[0]["candidate_rank"] == 1 for row in scored)


def test_candidate_scorer_training_cli_writes_model_manifest_and_summary(tmp_path: Path):
    out = tmp_path / "model"

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
            "20",
        ]
    )

    assert rc == 0
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert model["schema_version"] == "internal_sparse_candidate_scorer_v1"
    assert model["feature_input_policy"] == "inference_safe"
    assert summary["label_count"] == 3
    assert manifest["inference_stage"] == "sparse_candidate_scorer_training"
    assert manifest["feature_input_policy"] == "inference_safe"
    assert manifest["paper_safe_sparse_inference"] is True


def test_candidate_scorer_training_cli_accepts_solver_aware_feature_names(tmp_path: Path):
    out = tmp_path / "model"

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
            "--epochs",
            "20",
            "--feature_names",
            "native_score,negative_rank,valid,dense_consistent,sparse_inlier,negative_reprojection_error,solver_weight",
        ]
    )

    assert rc == 0
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert model["feature_names"] == [
        "native_score",
        "negative_rank",
        "valid",
        "dense_consistent",
        "sparse_inlier",
        "negative_reprojection_error",
        "solver_weight",
    ]
    assert summary["dense_teacher_sample_count"] == 4
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["feature_input_policy"] == "teacher_oracle_diagnostic"
    assert manifest["paper_safe_sparse_inference"] is False
    assert manifest["teacher_only_feature_names"] == [
        "dense_consistent",
        "sparse_inlier",
        "negative_reprojection_error",
        "solver_weight",
    ]
    assert manifest["hyperparameters"]["feature_names"] == model["feature_names"]
