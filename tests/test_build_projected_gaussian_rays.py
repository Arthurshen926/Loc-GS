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
                "query_xy": [10.0, 20.0],
                "source_xy": [10.0, 20.0],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_projections(path: Path, *, split_name: str = "selfmap_train") -> None:
    rows = [
        {"type": "manifest", "manifest": {"scene": "ToyScene", "split_name": split_name}},
        {
            "type": "projection",
            "projection": {
                "source_view_id": "s1",
                "split_name": split_name,
                "projection_source": "full_raw_gaussians",
                "gaussian_id": 0,
                "xy": [10.2, 20.1],
                "radius": 4.0,
                "opacity": 0.7,
                "depth": 5.0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_projected_gaussian_rays_cli_writes_source_rays(tmp_path):
    matches = tmp_path / "matches.jsonl"
    projections = tmp_path / "projections.jsonl"
    output_dir = tmp_path / "source_rays"
    _write_matches(matches)
    _write_projections(projections)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_projected_gaussian_rays",
            "--matches",
            str(matches),
            "--projections",
            str(projections),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ToyScene",
            "--split_name",
            "selfmap_train",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(completed.stdout)
    source_rays_path = output_dir / "source_rays.jsonl"
    assert payload["source_rays_path"] == str(source_rays_path)
    rows = [json.loads(line) for line in source_rays_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "manifest"
    assert rows[1]["type"] == "ray"
    assert rows[1]["ray"]["contributors"][0]["gaussian_id"] == 0
    assert json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))["ray_count"] == 1
    assert json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))["audit_status"] == "passed"


def test_build_projected_gaussian_rays_cli_rejects_test_split(tmp_path):
    matches = tmp_path / "matches.jsonl"
    projections = tmp_path / "projections.jsonl"
    _write_matches(matches, split_name="test")
    _write_projections(projections, split_name="test")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_projected_gaussian_rays",
            "--matches",
            str(matches),
            "--projections",
            str(projections),
            "--output_dir",
            str(tmp_path / "source_rays"),
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
