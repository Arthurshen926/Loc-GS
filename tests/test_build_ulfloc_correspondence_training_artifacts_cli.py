import json
import subprocess
import sys

import torch


def test_build_ulfloc_correspondence_training_artifacts_cli_writes_audit_bundle(tmp_path):
    supervision = {
        "schema_version": "correspondence_supervision_v1",
        "split_name": "train_self_map",
        "records": [
            {
                "query_id": "q.png",
                "gaussian_id": 1,
                "keypoint_xy": [2.0, 3.0],
                "descriptor_score": 0.9,
                "descriptor_margin": 0.3,
                "reprojection_error_px": 0.5,
                "local_geometry_score": 0.7,
                "label": 1,
                "pnp_inlier": True,
                "supervision_weight": 1.2,
            },
            {
                "query_id": "q.png",
                "gaussian_id": 4,
                "keypoint_xy": [8.0, 9.0],
                "descriptor_score": 0.88,
                "descriptor_margin": 0.01,
                "reprojection_error_px": 8.0,
                "local_geometry_score": 0.1,
                "label": 0,
                "pnp_inlier": False,
                "supervision_weight": 0.3,
            },
        ],
    }
    source = tmp_path / "correspondence_supervision.json"
    source.write_text(json.dumps(supervision), encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_ulfloc_correspondence_training_artifacts",
            "--correspondence_supervision",
            str(source),
            "--output_dir",
            str(output_dir),
            "--height",
            "12",
            "--width",
            "16",
            "--hard_negative_min_positive_distance_px",
            "8.0",
            "--solver_validity_power",
            "1.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "metrics_summary.json").exists()
    assert (output_dir / "split_audit.json").exists()
    assert (output_dir / "git_status.txt").exists()
    assert (output_dir / "pnp_ranking.json").exists()
    assert (output_dir / "descriptor_conflict_graph.json").exists()
    detector = torch.load(output_dir / "detector_targets.pt", map_location="cpu")
    scorer = torch.load(output_dir / "match_scorer.pt", map_location="cpu")
    assert detector["metadata"]["positive_detector_point_count"] == 1
    assert detector["metadata"]["hard_negative_detector_point_count"] == 1
    assert detector["metadata"]["detector_target_storage"] == "points"
    assert detector["targets"]["q.png"]["keypoint_yx"].shape == (1, 2)
    assert detector["targets"]["q.png"]["solver_validity_weights"].shape == (1,)
    assert detector["targets"]["q.png"]["negative_keypoint_yx"].shape == (1, 2)
    assert detector["targets"]["q.png"]["negative_weights"][0].item() > 0.0
    assert "heatmap" not in detector["targets"]["q.png"]
    assert scorer["labels"].tolist() == [1, 0]
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["conflict_edge_count"] == 1
    assert metrics["detector_target_storage"] == "points"
    assert metrics["detector_solver_validity_power"] == 1.0
    audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert audit["test_split_used"] is False
