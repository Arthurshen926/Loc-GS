import json
import subprocess
import sys
from pathlib import Path


def _write_matches(path: Path, *, split_name: str = "selfmap_train") -> None:
    rows = [
        {"type": "manifest", "manifest": {"scene": "ToyScene", "split_name": split_name}},
        {
            "type": "match",
            "match": {
                "query_id": "q1",
                "source_view_id": "s1",
                "split_name": split_name,
                "query_xy": [5.0, 6.0],
                "source_xy": [10.0, 20.0],
                "descriptor_score": 0.9,
                "local_geometry_score": 0.8,
                "pnp_inlier": True,
                "reprojection_error_px": 1.0,
                "expected_depth": 8.0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_source_rays(path: Path, *, split_name: str = "selfmap_train") -> None:
    rows = [
        {"type": "manifest", "manifest": {"scene": "ToyScene", "split_name": split_name}},
        {
            "type": "ray",
            "ray": {
                "source_view_id": "s1",
                "split_name": split_name,
                "pixel_xy": [10.0, 20.0],
                "rendered_depth": 3.0,
                "contributors": [
                    {"gaussian_id": 0, "contribution": 0.75, "depth": 3.0, "xyz": [0.0, 0.0, 3.0]},
                    {"gaussian_id": 1, "contribution": 0.25, "depth": 8.0, "xyz": [1.0, 0.0, 8.0]},
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_cross_view_ray_observations_cli_writes_audited_bundle(tmp_path):
    matches = tmp_path / "matches.jsonl"
    source_rays = tmp_path / "source_rays.jsonl"
    output_dir = tmp_path / "observations"
    _write_matches(matches)
    _write_source_rays(source_rays)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_cross_view_ray_observations",
            "--matches",
            str(matches),
            "--source_rays",
            str(source_rays),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ToyScene",
            "--split_name",
            "selfmap_train",
            "--radius_px",
            "1.0",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    observations_path = output_dir / "observations.jsonl"
    assert payload["observations_path"] == str(observations_path)
    rows = [json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "manifest"
    assert rows[1]["type"] == "observation"
    assert rows[1]["observation"]["contributors"][0]["gaussian_id"] == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ToyScene"
    assert manifest["split_name"] == "selfmap_train"
    assert metrics["observation_count"] == 1
    assert metrics["dropped_no_contributors"] == 0
    assert split_audit["audit_status"] == "passed"
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_cross_view_ray_observations_cli_rejects_test_split(tmp_path):
    matches = tmp_path / "matches.jsonl"
    source_rays = tmp_path / "source_rays.jsonl"
    _write_matches(matches, split_name="test")
    _write_source_rays(source_rays, split_name="test")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_cross_view_ray_observations",
            "--matches",
            str(matches),
            "--source_rays",
            str(source_rays),
            "--output_dir",
            str(tmp_path / "observations"),
            "--scene",
            "ToyScene",
            "--split_name",
            "test",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "test split" in completed.stderr
