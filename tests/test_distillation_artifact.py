import json
from pathlib import Path

import torch

from loc_gs.scripts.build_internal_distillation_artifact import main
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact
from loc_gs.teacher.distillation_artifact import (
    DistillationArtifactConfig,
    build_distillation_payload_from_observations,
    build_distillation_payload,
    load_teacher_observation_rows,
    load_solver_feedback_label_rows,
)


def _write_pair_cache(path: Path) -> Path:
    payload = {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        "cosine": torch.tensor([[0.95, 0.10], [0.90, 0.20]], dtype=torch.float32),
        "label": torch.tensor([1, 0], dtype=torch.int64),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[18.0, 1.0], [1.0, 20.0]], dtype=torch.float32),
        "query_id": ["q1.png::kp0", "q2.png::kp0"],
        "image_id": ["q1.png", "q2.png"],
        "keypoint_id": ["kp0", "kp0"],
        "source_phase": ["train", "train"],
    }
    torch.save(payload, path)
    return path


def _write_feedback(path: Path) -> Path:
    rows = [
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "query_id": "q1.png",
            "dense_helped": True,
            "distill_weight": 0.75,
        },
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "query_id": "q2.png",
            "dense_helped": False,
            "distill_weight": 0.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_teacher_observations(path: Path) -> Path:
    rows = [
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "image_id": "q1.png",
            "keypoint_id": "kp0",
            "candidate_rank": 1,
            "geometric_correct": True,
            "dense_consistent": True,
            "sparse_inlier": True,
            "reprojection_error_px": 1.0,
            "solver_weight": 3.0,
        },
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "image_id": "q1.png",
            "keypoint_id": "kp0",
            "candidate_rank": 0,
            "geometric_correct": False,
            "dense_consistent": False,
            "sparse_inlier": False,
            "reprojection_error_px": 18.0,
            "solver_weight": 1.5,
        },
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "image_id": "q2.png",
            "keypoint_id": "kp0",
            "landmark_id": 3,
            "geometric_correct": True,
            "dense_consistent": True,
            "sparse_inlier": True,
            "reprojection_error_px": 2.0,
            "solver_weight": 2.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_distillation_payload_merges_solver_feedback_into_candidate_grids(tmp_path: Path):
    payload = torch.load(_write_pair_cache(tmp_path / "pairs.pt"), map_location="cpu")
    labels = load_solver_feedback_label_rows(_write_feedback(tmp_path / "labels.jsonl"))

    distilled, summary = build_distillation_payload(
        payload,
        labels,
        scene="GreatCourt",
        split_name="train",
        cfg=DistillationArtifactConfig(dense_consistency_reprojection_px=4.0),
    )

    assert summary["feedback_query_count"] == 2
    assert summary["dense_helped_query_count"] == 1
    assert distilled["dense_consistent"].tolist() == [[False, True], [True, False]]
    assert distilled["sparse_inlier"].tolist() == [[False, True], [True, False]]
    assert distilled["label_roles"] == [["hard_negative", "protected_support"], ["protected_support", "hard_negative"]]
    assert distilled["solver_weight"].tolist()[0][1] > distilled["solver_weight"].tolist()[1][0]


def test_distillation_payload_overlays_per_candidate_teacher_observations(tmp_path: Path):
    payload = torch.load(_write_pair_cache(tmp_path / "pairs.pt"), map_location="cpu")
    observations = load_teacher_observation_rows(_write_teacher_observations(tmp_path / "observations.jsonl"))

    distilled, summary = build_distillation_payload_from_observations(
        payload,
        observations,
        scene="GreatCourt",
        split_name="train",
        cfg=DistillationArtifactConfig(max_solver_weight=4.0),
    )

    assert summary["observation_count"] == 3
    assert summary["matched_observation_count"] == 3
    assert summary["geometric_positive_row_count"] == 2
    assert distilled["label"].tolist() == [1, 0]
    assert distilled["dense_consistent"].tolist() == [[False, True], [True, False]]
    assert distilled["sparse_inlier"].tolist() == [[False, True], [True, False]]
    assert distilled["reprojection_error"].tolist() == [[18.0, 1.0], [2.0, float("inf")]]
    assert distilled["solver_weight"].tolist() == [[1.5, 3.0], [2.0, 1.0]]
    assert distilled["label_roles"] == [["hard_negative", "protected_support"], ["protected_support", "neutral"]]


def test_distillation_artifact_cli_writes_trainable_internal_artifact(tmp_path: Path):
    out = tmp_path / "distilled"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt")),
            "--solver_feedback_labels",
            str(_write_feedback(tmp_path / "labels.jsonl")),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    artifact = load_listwise_candidate_artifact(out / "distilled_candidates.pt")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert artifact.batches[0].candidate_dense_consistent == [[False, True]]
    assert artifact.batches[0].candidate_label_roles == [["hard_negative", "protected_support"]]
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["inference_stage"] == "distillation_artifact_generation"
    assert summary["candidate_row_count"] == 2


def test_distillation_artifact_cli_accepts_per_candidate_teacher_observations(tmp_path: Path):
    out = tmp_path / "distilled"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt")),
            "--teacher_observations",
            str(_write_teacher_observations(tmp_path / "observations.jsonl")),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    artifact = load_listwise_candidate_artifact(out / "distilled_candidates.pt")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert artifact.batches[0].candidate_dense_consistent == [[False, True]]
    assert artifact.batches[0].candidate_solver_weight == [[1.5, 3.0]]
    assert manifest["teacher_observations"].endswith("observations.jsonl")
    assert manifest["solver_feedback_labels"] is None
    assert summary["schema_version"] == "internal_observation_distillation_artifact_summary_v1"
    assert summary["matched_observation_count"] == 3
