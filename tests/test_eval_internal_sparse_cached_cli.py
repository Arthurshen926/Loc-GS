import json
import struct
from pathlib import Path

import numpy as np
import torch

from loc_gs.core.camera import load_camera_records
from loc_gs.core.geometry import project_points_w2c
from loc_gs.scripts.eval_internal_sparse_cached import main
from loc_gs.students.candidate_mlp_scorer import (
    CandidateMLPScorerConfig,
    build_candidate_mlp_feature_cache,
    train_candidate_mlp_scorer,
    train_candidate_mlp_scorer_from_feature_cache,
)
from loc_gs.students.descriptor_fusion import DescriptorFusionModel
from loc_gs.students.detector_student import DetectorStudentModel
from loc_gs.students.landmark_selector import LandmarkSelectorModel
from loc_gs.sparse.artifact_adapter import load_listwise_candidate_artifact


def _write_ply(path: Path) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 12\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    rows = [
        (-0.5, -0.4, 3.0),
        (0.5, -0.4, 3.1),
        (-0.4, 0.5, 2.9),
        (0.4, 0.5, 3.3),
        (0.0, 0.0, 2.6),
        (-0.7, 0.1, 3.4),
        (3.5, -0.4, 3.0),
        (4.5, -0.4, 3.1),
        (3.6, 0.5, 2.9),
        (4.4, 0.5, 3.3),
        (4.0, 0.0, 2.6),
        (3.3, 0.1, 3.4),
    ]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<3f", *row))
    return path


def _write_cameras(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "img_name": "img.png",
                    "width": 240,
                    "height": 180,
                    "fx": 140.0,
                    "fy": 140.0,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_pair_cache(path: Path, cameras: Path, *, buried_correct: bool = False, descriptor_signals: bool = False) -> Path:
    points = np.array(
        [
            (-0.5, -0.4, 3.0),
            (0.5, -0.4, 3.1),
            (-0.4, 0.5, 2.9),
            (0.4, 0.5, 3.3),
            (0.0, 0.0, 2.6),
            (-0.7, 0.1, 3.4),
        ],
        dtype=np.float64,
    )
    camera = load_camera_records(cameras, target_width=120, target_height=90, missing_principal_point="pixel_center")[
        "img.png"
    ]
    keypoints_xy, valid = project_points_w2c(points, camera.pose_w2c, camera.intrinsics)
    assert bool(valid.all())
    if buried_correct:
        landmark_id = torch.tensor(
            [[6, 0], [7, 1], [8, 2], [9, 3], [10, 4], [11, 5]],
            dtype=torch.int64,
        )
        cosine = torch.tensor([[0.95, 0.1]] * 6, dtype=torch.float32)
        label = torch.ones(6, dtype=torch.int64)
        topk = 2
        candidate_mask = torch.ones((6, 2), dtype=torch.bool)
    else:
        landmark_id = torch.arange(6, dtype=torch.int64).reshape(6, 1)
        cosine = torch.ones((6, 1), dtype=torch.float32)
        label = torch.zeros(6, dtype=torch.int64)
        topk = 1
        candidate_mask = torch.ones((6, 1), dtype=torch.bool)
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train_dev",
            "topk": topk,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(12 if buried_correct else 6, dtype=torch.int64),
        "query_yx": torch.tensor([[float(xy[1]), float(xy[0])] for xy in keypoints_xy], dtype=torch.float32),
        "landmark_id": landmark_id,
        "cosine": cosine,
        "label": label,
        "candidate_mask": candidate_mask,
        "reprojection_error": torch.where(
            torch.arange(topk, dtype=torch.int64).reshape(1, topk) == label.reshape(-1, 1),
            torch.zeros((6, topk), dtype=torch.float32),
            torch.full((6, topk), 1000.0, dtype=torch.float32),
        ),
        "query_id": [f"img.png::kp{i}" for i in range(6)],
        "image_id": ["img.png"] * 6,
        "keypoint_id": [f"kp{i}" for i in range(6)],
        "source_phase": ["train_dev"] * 6,
    }
    if descriptor_signals:
        payload["query_desc"] = torch.tensor([[1.0, 0.0]] * 6, dtype=torch.float32)
        payload["landmark_desc"] = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]] * 6, dtype=torch.float32)
        payload["dense_consistent"] = torch.tensor([[False, True]] * 6, dtype=torch.bool)
        payload["sparse_inlier"] = torch.tensor([[False, True]] * 6, dtype=torch.bool)
        payload["solver_weight"] = torch.tensor([[0.25, 3.0]] * 6, dtype=torch.float32)
        payload["label_roles"] = [["hard_negative", "protected_support"] for _idx in range(6)]
    torch.save(payload, path)
    return path


def test_eval_internal_sparse_cached_cli_writes_auditable_eval_bundle(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    out = tmp_path / "eval"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--output_dir",
            str(out),
            "--max_queries",
            "1",
            "--second_pnp_enabled",
            "--refine_with_inliers",
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    split_audit = json.loads((out / "split_audit.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "internal_sparse_cached_eval_metrics_v1"
    assert metrics["pose_metric_status"] == "verified"
    assert metrics["recall_10cm_5d"] == 1.0
    assert manifest["schema_version"] == "internal_sparse_cached_eval_manifest_v1"
    assert manifest["inference_stage"] == "sparse_only"
    assert manifest["dense_inference_enabled"] is False
    assert manifest["hyperparameters"]["second_pnp_enabled"] is True
    assert rows[0]["query_id"] == "img.png"
    assert split_audit["audit_status"] == "passed"
    assert (out / "command.txt").is_file()
    assert (out / "git_status.txt").is_file()


def test_eval_internal_sparse_cached_cli_auto_calibrates_camera_frame(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    out = tmp_path / "eval_auto_frame"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--output_dir",
            str(out),
            "--max_queries",
            "1",
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["frame_calibration_status"] == "passed"
    assert metrics["frame_calibration_best_frame"] == "auto_120x90_pixel_center"
    assert metrics["resolved_image_width"] == 120
    assert metrics["resolved_image_height"] == 90
    assert metrics["recall_10cm_5d"] == 1.0
    assert manifest["hyperparameters"]["frame_auto_calibration"] is True


def test_eval_internal_sparse_cached_cli_accepts_query_id_filter(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    query_ids = tmp_path / "query_ids.txt"
    query_ids.write_text("img.png\nmissing.png\n", encoding="utf-8")
    out = tmp_path / "eval_filtered"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--query_ids",
            str(query_ids),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["query_filter_enabled"] is True
    assert metrics["requested_query_count"] == 2
    assert metrics["matched_query_count"] == 1
    assert metrics["missing_query_count"] == 1
    assert metrics["missing_query_ids_preview"] == ["missing.png"]


def test_eval_internal_sparse_cached_cli_accepts_landmark_selector(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    selector_path = tmp_path / "landmark_selector.json"
    selector = LandmarkSelectorModel(
        landmark_scores={str(idx): 3.0 for idx in range(6)},
        conflict_edges={},
        conflict_degrees={},
        score_scale=1.0,
    )
    selector_path.write_text(json.dumps(selector.to_json_dict(), sort_keys=True), encoding="utf-8")
    out = tmp_path / "eval_selector"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras, buried_correct=True)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--landmark_selector",
            str(selector_path),
            "--rerank_prefix_fraction",
            "0",
            "--native_weight",
            "0",
            "--solver_weight",
            "1",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert metrics["landmark_selector_enabled"] is True
    assert rows[0]["success"] is True
    assert rows[0]["te_cm"] < 1.0


def test_eval_internal_sparse_cached_cli_accepts_descriptor_fusion(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    fusion_path = tmp_path / "descriptor_fusion.pt"
    fusion = DescriptorFusionModel(
        fused_descriptors={str(idx): [1.0, 0.0] for idx in range(6)},
        landmark_scores={str(idx): 3.0 for idx in range(6)},
        negative_scores={},
        score_scale=1.0,
    )
    torch.save(fusion.to_torch_dict(), fusion_path)
    out = tmp_path / "eval_fusion"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras, buried_correct=True)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--descriptor_fusion",
            str(fusion_path),
            "--rerank_prefix_fraction",
            "0",
            "--native_weight",
            "0",
            "--solver_weight",
            "1",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert metrics["descriptor_fusion_enabled"] is True
    assert rows[0]["success"] is True
    assert rows[0]["te_cm"] < 1.0


def test_eval_internal_sparse_cached_cli_accepts_mlp_candidate_scorer(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    pairs = _write_pair_cache(tmp_path / "pairs.pt", cameras, buried_correct=True, descriptor_signals=True)
    artifact = load_listwise_candidate_artifact(pairs)
    model, summary = train_candidate_mlp_scorer(
        artifact,
        CandidateMLPScorerConfig(epochs=100, learning_rate=0.03, hidden_dim=8, seed=11),
    )
    assert summary["trained_top1_correct"] == 6
    scorer_path = tmp_path / "candidate_mlp.pt"
    torch.save(model.to_torch_dict(), scorer_path)
    out = tmp_path / "eval_mlp_scorer"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(pairs),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--candidate_scorer",
            str(scorer_path),
            "--rerank_prefix_fraction",
            "0",
            "--native_weight",
            "0",
            "--solver_weight",
            "1",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert metrics["candidate_scorer_enabled"] is True
    assert metrics["rerank_diagnostic_enabled"] is True
    assert metrics["native_top1_correct"] == 0
    assert metrics["reranked_top1_correct"] == 6
    assert metrics["reranked_top1_gain"] == 6
    assert metrics["selected_geometric_correct_count_median"] == 6
    assert metrics["selected_geometric_correct_ratio_median"] == 1.0
    assert metrics["selected_keypoint_bbox_area_fraction_median"] > 0.01
    assert metrics["selected_depth_range_m_median"] > 0.0
    assert rows[0]["rerank_diagnostic"]["native_top1_correct"] == 0
    assert rows[0]["rerank_diagnostic"]["reranked_top1_correct"] == 6
    assert rows[0]["rerank_diagnostic"]["reranked_top1_gain"] == 6
    assert rows[0]["selected_set_diagnostics"]["selected_geometric_correct_count"] == 6
    assert rows[0]["selected_set_diagnostics"]["selected_geometric_correct_ratio"] == 1.0
    assert rows[0]["success"] is True
    assert rows[0]["te_cm"] < 1.0


def test_eval_internal_sparse_cached_cli_accepts_cache_trained_mlp_candidate_scorer(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    pairs = _write_pair_cache(tmp_path / "pairs.pt", cameras, buried_correct=True, descriptor_signals=True)
    artifact = load_listwise_candidate_artifact(pairs)
    cfg = CandidateMLPScorerConfig(epochs=100, learning_rate=0.03, hidden_dim=8, seed=11)
    feature_cache = build_candidate_mlp_feature_cache(artifact, cfg)
    model, summary = train_candidate_mlp_scorer_from_feature_cache(feature_cache, cfg)
    assert summary["feature_materialization"] == "feature_cache"
    assert summary["trained_top1_correct"] == 6
    scorer_path = tmp_path / "candidate_mlp_from_cache.pt"
    torch.save(model.to_torch_dict(), scorer_path)
    out = tmp_path / "eval_mlp_scorer_from_cache"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(pairs),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--candidate_scorer",
            str(scorer_path),
            "--rerank_prefix_fraction",
            "0",
            "--native_weight",
            "0",
            "--solver_weight",
            "1",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert manifest["dense_inference_enabled"] is False
    assert manifest["inference_stage"] == "sparse_only"
    assert metrics["candidate_scorer_enabled"] is True
    assert rows[0]["success"] is True
    assert rows[0]["te_cm"] < 1.0


def test_eval_internal_sparse_cached_cli_accepts_detector_student(tmp_path: Path):
    cameras = _write_cameras(tmp_path / "cameras.json")
    detector_path = tmp_path / "detector_student.json"
    detector = DetectorStudentModel(
        grid_size=1,
        x_extent=120.0,
        y_extent=90.0,
        cell_scores={"0:0": 3.0},
        score_scale=1.0,
    )
    detector_path.write_text(json.dumps(detector.to_json_dict(), sort_keys=True), encoding="utf-8")
    out = tmp_path / "eval_detector"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
            "--candidate_artifact",
            str(_write_pair_cache(tmp_path / "pairs.pt", cameras, buried_correct=True)),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--image_width",
            "120",
            "--image_height",
            "90",
            "--detector_student",
            str(detector_path),
            "--rerank_prefix_fraction",
            "0",
            "--native_weight",
            "0",
            "--solver_weight",
            "1",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["detector_student_enabled"] is True
