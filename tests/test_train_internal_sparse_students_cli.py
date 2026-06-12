import json
from pathlib import Path

import torch

from loc_gs.scripts.train_internal_sparse_students import main
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact


def _write_cameras(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {"img_name": "a.png", "width": 100, "height": 80, "fx": 50.0, "fy": 50.0},
                {"img_name": "b.png", "width": 100, "height": 80, "fx": 50.0, "fy": 50.0},
                {"img_name": "c.png", "width": 100, "height": 80, "fx": 50.0, "fy": 50.0},
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_pair_cache(path: Path) -> Path:
    payload = {
        "metadata": {"format": "listwise", "scene": "GreatCourt", "source_split_name": "train", "topk": 2},
        "query_desc": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        "landmark_desc": torch.tensor(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "query_yx": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        "landmark_id": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int64),
        "cosine": torch.tensor([[0.95, 0.10], [0.90, 0.20], [0.85, 0.30], [0.80, 0.40]], dtype=torch.float32),
        "label": torch.tensor([1, 1, 0, 0], dtype=torch.int64),
        "candidate_mask": torch.ones((4, 2), dtype=torch.bool),
        "reprojection_error": torch.tensor([[18.0, 1.0], [16.0, 1.5], [1.0, 20.0], [1.0, 22.0]], dtype=torch.float32),
        "query_id": ["a.png::kp0", "a.png::kp1", "b.png::kp0", "b.png::kp1"],
        "image_id": ["a.png", "a.png", "b.png", "b.png"],
        "keypoint_id": ["kp0", "kp1", "kp0", "kp1"],
        "source_phase": ["train", "train", "train", "train"],
    }
    torch.save(payload, path)
    return path


def _write_feedback(path: Path) -> Path:
    rows = [
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "query_id": "a.png",
            "dense_helped": True,
            "distill_weight": 0.75,
        },
        {
            "scene": "GreatCourt",
            "split_name": "train",
            "query_id": "b.png",
            "dense_helped": False,
            "distill_weight": 0.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_inlier_precision_feedback(path: Path) -> Path:
    payload = {
        "schema_version": "internal_inlier_precision_feedback_v1",
        "scene": "GreatCourt",
        "split_name": "train",
        "query_count": 2,
        "hard_query_count": 1,
        "set_selection_hard_count": 1,
        "inlier_precision_hard_count": 1,
        "post_pnp_rescore_harm_count": 0,
        "per_query": [
            {
                "schema_version": "internal_inlier_precision_feedback_row_v1",
                "scene": "GreatCourt",
                "split_name": "train",
                "query_id": "a.png",
                "query_role": "set_and_inlier_precision_hard",
                "selected_set_hard": True,
                "inlier_precision_hard": True,
                "post_pnp_rescore_harm": False,
                "selected_geometric_correct_ratio": 0.05,
                "inlier_geometric_correct_ratio": 0.20,
                "selected_gap": 0.5,
                "inlier_gap": 0.6,
                "set_selection_weight": 1.8,
                "hard_negative_weight": 2.4,
                "scorer_distill_weight": 2.2,
            },
            {
                "schema_version": "internal_inlier_precision_feedback_row_v1",
                "scene": "GreatCourt",
                "split_name": "train",
                "query_id": "b.png",
                "query_role": "stable",
                "selected_set_hard": False,
                "inlier_precision_hard": False,
                "post_pnp_rescore_harm": False,
                "selected_geometric_correct_ratio": 0.35,
                "inlier_geometric_correct_ratio": 0.80,
                "selected_gap": 0.0,
                "inlier_gap": 0.0,
                "set_selection_weight": 1.0,
                "hard_negative_weight": 1.0,
                "scorer_distill_weight": 0.0,
            },
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_render_manifest(path: Path) -> Path:
    rows = [
        {
            "synthetic_query_id": f"sim/GreatCourt/train/{idx:06d}",
            "rgb_path": f"/renders/{idx:06d}.png",
            "rgb_exists": True,
        }
        for idx in range(4)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_train_internal_sparse_students_cli_writes_online_training_bundle(tmp_path: Path):
    out = tmp_path / "train"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--cameras_json",
            str(_write_cameras(tmp_path / "cameras.json")),
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt")),
            "--solver_feedback_labels",
            str(_write_feedback(tmp_path / "labels.jsonl")),
            "--inlier_precision_feedback",
            str(_write_inlier_precision_feedback(tmp_path / "inlier_precision_feedback.json")),
            "--render_manifest",
            str(_write_render_manifest(tmp_path / "render_manifest.jsonl")),
            "--output_dir",
            str(out),
            "--sample_count",
            "4",
            "--seed",
            "3",
            "--max_keypoints_per_episode",
            "1",
            "--epochs",
            "20",
            "--candidate_mlp_batch_size",
            "2",
            "--candidate_mlp_cache_features",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    candidate_mlp = torch.load(out / "candidate_mlp_scorer.pt", map_location="cpu")
    candidate_mlp_cache = torch.load(out / "candidate_mlp_feature_cache.pt", map_location="cpu")
    selector = json.loads((out / "landmark_selector.json").read_text(encoding="utf-8"))
    conflicts = json.loads((out / "conflict_graph.json").read_text(encoding="utf-8"))
    fusion = json.loads((out / "descriptor_fusion.json").read_text(encoding="utf-8"))
    detector = json.loads((out / "detector_student.json").read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in (out / "online_episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    artifact = load_listwise_candidate_artifact(out / "online_distilled_candidates.pt")

    assert manifest["inference_stage"] == "online_sparse_student_training"
    assert manifest["dense_teacher_enabled"] is True
    assert manifest["dense_inference_enabled"] is False
    assert manifest["external_runtime_dependency"] == "forbidden"
    assert manifest["inlier_precision_feedback"] == str(tmp_path / "inlier_precision_feedback.json")
    assert manifest["render_manifest"] == str(tmp_path / "render_manifest.jsonl")
    assert manifest["hyperparameters"]["require_rendered_rgb"] is False
    assert summary["online_episode_count"] == 4
    assert summary["missing_candidate_count"] == 0
    assert summary["render_ready_episode_count"] == 4
    assert summary["missing_render_episode_count"] == 0
    assert manifest["hyperparameters"]["camera_sampling_source"] == "candidate_artifact_sources"
    assert summary["student_modules"] == [
        "correspondence_scorer",
        "candidate_mlp_scorer",
        "landmark_selector",
        "conflict_graph",
        "descriptor_fusion",
        "detector_student",
    ]
    assert model["schema_version"] == "internal_sparse_candidate_scorer_v1"
    assert candidate_mlp["schema_version"] == "internal_candidate_mlp_scorer_v1"
    assert candidate_mlp["score_calibration"] == "train_logit_zscore"
    assert summary["candidate_mlp_scorer"]["student_modules"] == ["candidate_mlp_scorer"]
    assert summary["candidate_mlp_scorer"]["batch_size"] == 2
    assert summary["candidate_mlp_scorer"]["feature_materialization"] == "feature_cache"
    assert summary["candidate_mlp_scorer"]["feature_cache_enabled"] is True
    assert summary["candidate_mlp_feature_cache"]["schema_version"] == "internal_candidate_mlp_feature_cache_summary_v1"
    assert summary["inlier_precision_feedback"]["schema_version"] == "internal_inlier_precision_feedback_merge_summary_v1"
    assert summary["inlier_precision_feedback"]["boosted_solver_feedback_count"] == 1
    assert candidate_mlp_cache["schema_version"] == "internal_candidate_mlp_feature_cache_v1"
    assert manifest["candidate_mlp_feature_cache"] == str(out / "candidate_mlp_feature_cache.pt")
    assert manifest["hyperparameters"]["candidate_mlp_batch_size"] == 2
    assert manifest["hyperparameters"]["candidate_mlp_stream_features"] is False
    assert manifest["hyperparameters"]["candidate_mlp_cache_features"] is True
    assert selector["schema_version"] == "internal_landmark_selector_v1"
    assert conflicts["schema_version"] == "internal_conflict_graph_v1"
    assert fusion["schema_version"] == "internal_descriptor_fusion_v1"
    assert detector["schema_version"] == "internal_detector_student_v1"
    assert episodes[0]["schema_version"] == "internal_online_sparse_dense_episode_v1"
    assert artifact.keypoint_count <= 4
