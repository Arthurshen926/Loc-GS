import json
import subprocess
import sys
from pathlib import Path
from collections.abc import Iterator

import pytest
import torch

from loc_gs.feedback.rendered_feedback_augmentation import (
    SCHEMA_VERSION,
    accumulate_rendered_feedback,
)


def test_observed_only_accumulates_existing_observation_records_without_rendered_fields():
    observations = [
        {
            "query_id": "train/q1",
            "source_view_id": "train/s1",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 2.0,
            "descriptor_score": 0.8,
            "local_geometry_score": 1.0,
            "contributors": [
                {"gaussian_id": 0, "contribution": 0.25},
                {"gaussian_id": 1, "contribution": 0.75},
            ],
        }
    ]

    artifact = accumulate_rendered_feedback(
        observations,
        num_gaussians=3,
        split_name="selfmap_train",
        mode="observed_only",
    )

    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["observed_count"].tolist() == [1, 1, 0]
    assert artifact["positive_observed_count"].tolist() == [1, 1, 0]
    assert artifact["support_score"][1] > artifact["support_score"][0] > 0.0
    assert artifact["artifact_risk"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert artifact["metadata"]["mode"] == "observed_only"
    assert artifact["metadata"]["split_name"] == "selfmap_train"


def test_rendered_depth_augmented_attributes_artifact_risk_to_near_contributors():
    observations = [
        {
            "query_id": "train/q1",
            "source_view_id": "train/s1",
            "split_name": "selfmap_train",
            "pnp_inlier": True,
            "reprojection_error_px": 1.0,
            "descriptor_score": 1.0,
            "local_geometry_score": 1.0,
            "rendered_depth": 2.0,
            "expected_depth": 10.0,
            "artifact_score": 0.9,
            "ray_contributors": [
                {"gaussian_id": 0, "weight": 0.8, "depth": 2.0},
                {"gaussian_id": 1, "weight": 0.2, "depth": 10.0},
            ],
        }
    ]

    artifact = accumulate_rendered_feedback(
        observations,
        num_gaussians=3,
        split_name="selfmap_train",
        mode="rendered_depth_augmented",
        depth_margin=1.0,
    )

    assert artifact["support_score"][0] > artifact["support_score"][1] > 0.0
    assert artifact["artifact_risk"][0] > 0.0
    assert artifact["artifact_risk"][1] == pytest.approx(0.0)
    assert artifact["artifact_risk_contribution"][0] > artifact["artifact_risk_contribution"][1]
    assert artifact["metadata"]["rendered_observation_count"] == 1
    assert artifact["metadata"]["mode"] == "rendered_depth_augmented"


def test_accumulate_rendered_feedback_rejects_test_split():
    with pytest.raises(ValueError, match="test split"):
        accumulate_rendered_feedback([], num_gaussians=1, split_name="test", mode="observed_only")

    with pytest.raises(ValueError, match="test split"):
        accumulate_rendered_feedback(
            [{"split_name": "test", "contributors": [{"gaussian_id": 0, "weight": 1.0}]}],
            num_gaussians=1,
            split_name="selfmap_train",
            mode="rendered_depth_augmented",
        )


def _write_observations(path: Path, *, split_name: str = "selfmap_train") -> None:
    rows = [
        {
            "type": "manifest",
            "manifest": {
                "scene": "ToyScene",
                "split_name": split_name,
                "checkpoint_path": "/tmp/checkpoint.pt",
                "map_path": "/tmp/map",
                "data_root": "/tmp/data",
            },
        },
        {
            "type": "observation",
            "observation": {
                "query_id": "train/q1",
                "source_view_id": "train/s1",
                "split_name": split_name,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "descriptor_score": 1.0,
                "rendered_depth": 2.0,
                "expected_depth": 8.0,
                "ray_contributors": [
                    {"gaussian_id": 0, "weight": 0.7, "depth": 2.0},
                    {"gaussian_id": 1, "weight": 0.3, "depth": 8.0},
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_rendered_feedback_observations_cli_writes_audited_artifact(tmp_path):
    observations = tmp_path / "observations.jsonl"
    output_dir = tmp_path / "rendered_feedback"
    _write_observations(observations)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_rendered_feedback_observations",
            "--observations",
            str(observations),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ToyScene",
            "--split_name",
            "selfmap_train",
            "--num_gaussians",
            "3",
            "--mode",
            "rendered_depth_augmented",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    artifact_path = output_dir / "rendered_feedback_support.pt"
    assert payload["artifact_path"] == str(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu")
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["artifact_risk"][0] > artifact["artifact_risk"][1]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ToyScene"
    assert manifest["split_name"] == "selfmap_train"
    assert manifest["checkpoint_path"] == "/tmp/checkpoint.pt"
    assert manifest["map_path"] == "/tmp/map"
    assert manifest["data_roots"] == ["/tmp/data"]
    assert manifest["feedback_flags"]["residual_feedback_enabled"] is False
    assert manifest["feedback_flags"]["selector_feedback_enabled"] is False
    assert manifest["feedback_flags"]["rho_feedback_enabled"] is False
    assert metrics["observed_gaussians"] == 2
    assert split_audit["audit_status"] == "passed"
    assert split_audit["test_split_used"] is False
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_rendered_feedback_observations_cli_rejects_test_split(tmp_path):
    observations = tmp_path / "observations.jsonl"
    _write_observations(observations, split_name="test")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_rendered_feedback_observations",
            "--observations",
            str(observations),
            "--output_dir",
            str(tmp_path / "rendered_feedback"),
            "--scene",
            "ToyScene",
            "--split_name",
            "test",
            "--num_gaussians",
            "3",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "test split" in completed.stderr


def test_load_observations_streams_jsonl_records(tmp_path):
    observations = tmp_path / "observations.jsonl"
    _write_observations(observations)

    from loc_gs.scripts.build_rendered_feedback_observations import load_observations

    manifest, records = load_observations(observations)

    assert manifest["scene"] == "ToyScene"
    assert isinstance(records, Iterator)
    assert next(records)["query_id"] == "train/q1"
