from __future__ import annotations

import json
from pathlib import Path

from loc_gs.scripts.build_sparse_failure_decomposition import build_argparser, main
from loc_gs.feedback.io import save_feedback_bank


def test_build_sparse_failure_decomposition_cli_writes_query_reports(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            [
                {
                    "sparse": {"inliers": 3, "inlier_rate": 0.02},
                    "sparse_TE": 500.0,
                    "dense": [{"dense_stats": {"valid_dense_match_ratio": 0.2, "artifact_depth_risk_score": 0.8}}],
                },
                {
                    "sparse": {"inliers": 40, "inlier_rate": 0.2},
                    "sparse_TE": 12.0,
                    "dense": [{"dense_stats": {"valid_dense_match_ratio": 0.9}}],
                },
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = build_argparser().parse_args(
        [
            "--results_json",
            str(results),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_dev",
        ]
    )

    assert main(args) == 0

    report = json.loads((output_dir / "decomposition.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))

    assert len(report["queries"]) == 2
    assert report["queries"][0]["primary_cause"] == "map_or_render_artifact"
    assert metrics["query_count"] == 2
    assert metrics["cause_counts"]["map_or_render_artifact"] == 1
    assert (output_dir / "decomposition.csv").exists()


def test_build_sparse_failure_decomposition_cli_reads_feedback_bank(tmp_path: Path) -> None:
    bank = tmp_path / "feedback_bank.jsonl"
    records = []
    for idx in range(40):
        records.append(
            {
                "scene": "ShopFacade",
                "query_id": "q0",
                "matched_gaussian_id": str(idx),
                "query_xy_norm": [0.1 + 0.01 * idx, 0.2],
                "descriptor_score": 0.95 - idx * 0.001,
                "descriptor_margin": 0.01,
                "pnp_inlier": idx > 30,
                "reprojection_error_px": 1.0 if idx > 30 else 20.0,
                "depth_m": 4.0 + idx * 0.1,
            }
        )
    save_feedback_bank(bank, records, {"scene": "ShopFacade", "split_name": "train_dev"})
    output_dir = tmp_path / "feedback_out"
    args = build_argparser().parse_args(
        [
            "--feedback_bank",
            str(bank),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ShopFacade",
            "--split_name",
            "train_dev",
        ]
    )

    assert main(args) == 0

    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "decomposition.json").read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert report["queries"][0]["primary_cause"] in {"descriptor_ambiguity", "ranking_failure"}
