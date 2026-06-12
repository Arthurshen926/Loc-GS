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


def _add_rerank_diagnostic(path: Path) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        {
            "schema_version": "internal_sparse_cached_eval_metrics_v1",
            "rerank_diagnostic_enabled": True,
            "rerank_diagnostic_query_count": 1536,
            "native_top1_correct": 91,
            "reranked_top1_correct": 225,
            "reranked_top1_gain": 134,
            "reranked_top1_changed_count": 1065,
            "reranked_topk_available": 259,
        }
    )
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _add_selected_set_diagnostic(path: Path) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        {
            "schema_version": "internal_sparse_cached_eval_metrics_v1",
            "selected_geometric_correct_count_median": 32.0,
            "selected_geometric_correct_ratio_median": 0.0625,
            "selected_keypoint_bbox_area_fraction_median": 0.9497,
            "selected_depth_range_m_median": 46.5,
        }
    )
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _add_inlier_set_diagnostic(path: Path) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        {
            "schema_version": "internal_sparse_cached_eval_metrics_v1",
            "inlier_geometric_correct_count_median": 21.0,
            "inlier_geometric_correct_ratio_median": 0.18,
            "inlier_keypoint_bbox_area_fraction_median": 0.75,
            "inlier_depth_range_m_median": 35.0,
        }
    )
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _add_post_pnp_rescore_diagnostic(path: Path) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        {
            "schema_version": "internal_sparse_cached_eval_metrics_v1",
            "post_pnp_candidate_rescore_enabled": True,
            "post_pnp_rescore_changed_count_median": 44.0,
            "post_pnp_rescore_corrected_count_median": 12.0,
            "post_pnp_rescore_worsened_count_median": 3.0,
            "post_pnp_rescore_correct_delta_median": 9.0,
        }
    )
    path.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_coverage_metrics(path: Path, *, complete: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "internal_candidate_coverage_v1",
                "scene": "GreatCourt",
                "split_name": "train_dev_seed13_20p",
                "candidate_artifact": "pairs.pt",
                "requested_query_count": 7,
                "covered_query_count": 7 if complete else 3,
                "missing_query_count": 0 if complete else 4,
                "coverage_ratio": 1.0 if complete else 3.0 / 7.0,
                "complete_coverage": complete,
                "coverage_status": "complete" if complete else "partial",
                "missing_query_ids_preview": [] if complete else ["missing_a.png"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_candidate_scorer_training_summary(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "internal_candidate_mlp_scorer_training_summary_v1",
                "student_modules": ["candidate_mlp_scorer"],
                "feature_materialization": "feature_cache",
                "feature_input_policy": "inference_safe",
                "paper_safe_sparse_inference": True,
                "sample_count": 40000,
                "label_count": 1073,
                "native_top1_correct": 294,
                "trained_top1_correct": 573,
                "dense_teacher_sample_count": 40000,
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


def test_internal_sparse_gate_includes_candidate_scorer_training_evidence(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _write_metrics(tmp_path / "candidate.json", median_te_cm=13.0, split_name="train_dev_seed13_20p")
    scorer_summary = _write_candidate_scorer_training_summary(tmp_path / "scorer_metrics.json")
    out = tmp_path / "gate_scorer"

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
            "--candidate_scorer_metrics",
            str(scorer_summary),
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    evidence = metrics["candidate_scorer_training"]
    assert manifest["candidate_scorer_metrics"] == str(scorer_summary)
    assert evidence["schema_version"] == "internal_candidate_mlp_scorer_training_summary_v1"
    assert evidence["feature_materialization"] == "feature_cache"
    assert evidence["paper_safe_sparse_inference"] is True
    assert evidence["native_top1_correct"] == 294
    assert evidence["trained_top1_correct"] == 573
    assert evidence["top1_gain"] == 279
    assert evidence["relative_top1_gain"] == 279 / 294


def test_internal_sparse_gate_includes_candidate_eval_rerank_diagnostic(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _add_rerank_diagnostic(
        _write_metrics(tmp_path / "candidate.json", median_te_cm=13.0, split_name="train_dev_seed13_20p")
    )
    out = tmp_path / "gate_eval_rerank"

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
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["candidate_rerank_diagnostic"] == {
        "native_top1_correct": 91,
        "rerank_diagnostic_enabled": True,
        "rerank_diagnostic_query_count": 1536,
        "reranked_top1_changed_count": 1065,
        "reranked_top1_correct": 225,
        "reranked_top1_gain": 134,
        "reranked_topk_available": 259,
    }


def test_internal_sparse_gate_includes_candidate_selected_set_diagnostic(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _add_selected_set_diagnostic(
        _write_metrics(tmp_path / "candidate.json", median_te_cm=13.0, split_name="train_dev_seed13_20p")
    )
    out = tmp_path / "gate_eval_selected_set"

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
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["candidate_selected_set_diagnostic"] == {
        "selected_depth_range_m_median": 46.5,
        "selected_geometric_correct_count_median": 32.0,
        "selected_geometric_correct_ratio_median": 0.0625,
        "selected_keypoint_bbox_area_fraction_median": 0.9497,
    }


def test_internal_sparse_gate_includes_candidate_inlier_set_diagnostic(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _add_inlier_set_diagnostic(
        _write_metrics(tmp_path / "candidate.json", median_te_cm=13.0, split_name="train_dev_seed13_20p")
    )
    out = tmp_path / "gate_eval_inlier_set"

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
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["candidate_inlier_set_diagnostic"] == {
        "inlier_depth_range_m_median": 35.0,
        "inlier_geometric_correct_count_median": 21.0,
        "inlier_geometric_correct_ratio_median": 0.18,
        "inlier_keypoint_bbox_area_fraction_median": 0.75,
    }


def test_internal_sparse_gate_includes_candidate_post_pnp_rescore_diagnostic(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _add_post_pnp_rescore_diagnostic(
        _write_metrics(tmp_path / "candidate.json", median_te_cm=13.0, split_name="train_dev_seed13_20p")
    )
    out = tmp_path / "gate_eval_post_pnp"

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
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["candidate_post_pnp_rescore_diagnostic"] == {
        "post_pnp_candidate_rescore_enabled": True,
        "post_pnp_rescore_changed_count_median": 44.0,
        "post_pnp_rescore_correct_delta_median": 9.0,
        "post_pnp_rescore_corrected_count_median": 12.0,
        "post_pnp_rescore_worsened_count_median": 3.0,
    }


def test_internal_sparse_gate_blocks_pass_when_candidate_coverage_is_incomplete(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev_seed13_20p")
    candidate = _write_metrics(tmp_path / "candidate.json", median_te_cm=9.0, split_name="train_dev_seed13_20p")
    coverage = _write_coverage_metrics(tmp_path / "coverage.json", complete=False)
    out = tmp_path / "gate_coverage"

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
            "--candidate_coverage_metrics",
            str(coverage),
            "--output_dir",
            str(out),
            "--dense_target_cm",
            "10.0",
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["sparse_gate_status"] == "blocked_candidate_coverage_incomplete"
    assert metrics["candidate_coverage"]["complete_coverage"] is False
    assert metrics["candidate_coverage"]["covered_query_count"] == 3
    assert metrics["candidate_coverage"]["missing_query_count"] == 4
    assert manifest["candidate_coverage_metrics"] == str(coverage)


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


def test_internal_sparse_gate_marks_split_or_query_mismatch_diagnostic(tmp_path: Path):
    pair_cache = _write_pair_cache(tmp_path / "pairs.pt")
    baseline = _write_metrics(tmp_path / "baseline.json", median_te_cm=15.0, split_name="train_dev")
    candidate = _write_metrics(tmp_path / "candidate.json", median_te_cm=9.0, split_name="train")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    data["schema_version"] = "internal_sparse_cached_eval_metrics_v1"
    data["pose_metric_status"] = "verified"
    data["query_count"] = 2
    candidate.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "gate_mismatch"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train_dev",
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
    assert metrics["sparse_gate_status"] == "diagnostic_split_or_query_mismatch"
    assert metrics["baseline_split_name"] == "train_dev"
    assert metrics["candidate_split_name"] == "train"
    assert metrics["baseline_query_count"] == 7
    assert metrics["candidate_query_count"] == 2
