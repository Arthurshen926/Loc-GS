import json

import torch

from loc_gs.scripts.build_query_view_validation_pruning import main


def test_build_query_view_validation_pruning_cli_writes_pruned_plan_and_audit(tmp_path):
    active_plan = tmp_path / "active_plan.json"
    active_plan.write_text(
        json.dumps(
            {
                "split_name": "train_dev",
                "landmark_fusion_plan": {
                    "10": {"selected_view_ids": ["view_a"]},
                    "11": {"selected_view_ids": ["view_b"]},
                },
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "results.json").write_text(
        json.dumps({"rows": [{"image_name": "q0", "sparse_te_cm": 2.0}]}),
        encoding="utf-8",
    )
    (candidate / "results.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "image_name": "q0",
                        "sparse_te_cm": 35.0,
                        "sparse_match_attributions": [
                            {
                                "matched_gaussian_id": 10,
                                "pnp_inlier": False,
                                "descriptor_score": 1.0,
                                "reprojection_error_px": 16.0,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    rc = main(
        [
            "--active_fusion_plan",
            str(active_plan),
            "--baseline_run_dir",
            str(baseline),
            "--candidate_run_dir",
            str(candidate),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_dev",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    pruned = json.loads((out / "pruned_landmark_fusion_plan.json").read_text())
    assert "10" not in pruned["landmark_fusion_plan"]
    assert "11" in pruned["landmark_fusion_plan"]
    metrics = json.loads((out / "metrics_summary.json").read_text())
    assert metrics["removed_pair_count"] == 1
    assert metrics["attribution_status"] == "attributed"
    split_audit = json.loads((out / "split_audit.json").read_text())
    assert split_audit["official_test_used"] is False
    assert split_audit["test_split_used"] is False
    assert (out / "manifest.json").exists()
    assert (out / "command.txt").exists()
    assert (out / "git_status.txt").exists()


def test_build_query_view_validation_pruning_cli_can_use_trace_payload(tmp_path):
    active_plan = tmp_path / "active_plan.json"
    active_plan.write_text(
        json.dumps(
            {
                "split_name": "train_selfmap",
                "landmark_fusion_plan": {
                    "10": {"selected_view_ids": ["view_a"]},
                    "11": {"selected_view_ids": ["view_b"]},
                },
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "results.json").write_text(
        json.dumps({"rows": [{"image_name": "q0", "sparse_te_cm": 2.0}]}),
        encoding="utf-8",
    )
    (candidate / "results.json").write_text(
        json.dumps({"rows": [{"image_name": "q0", "sparse_te_cm": 35.0}]}),
        encoding="utf-8",
    )
    trace = tmp_path / "trace.pt"
    torch.save(
        {
            "split_name": "train_selfmap",
            "correspondences": [
                {
                    "query_id": "q0",
                    "matched_gaussian_id": 10,
                    "pnp_inlier": False,
                    "descriptor_score": 1.0,
                    "reprojection_error_px": 16.0,
                }
            ],
        },
        trace,
    )
    out = tmp_path / "out"

    rc = main(
        [
            "--active_fusion_plan",
            str(active_plan),
            "--baseline_run_dir",
            str(baseline),
            "--candidate_run_dir",
            str(candidate),
            "--candidate_trace_payload",
            str(trace),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_selfmap",
            "--output_dir",
            str(out),
        ]
    )

    assert rc == 0
    pruned = json.loads((out / "pruned_landmark_fusion_plan.json").read_text())
    assert "10" not in pruned["landmark_fusion_plan"]
    metrics = json.loads((out / "metrics_summary.json").read_text())
    assert metrics["candidate_trace_payload"] == str(trace)
    assert metrics["attribution_status"] == "attributed"
