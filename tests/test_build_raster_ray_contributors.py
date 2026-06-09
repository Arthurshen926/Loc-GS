import json
import subprocess
import sys
from pathlib import Path


def _write_intersections(path: Path, *, split_name: str = "selfmap_train") -> None:
    rows = [
        {"type": "manifest", "manifest": {"scene": "ToyScene", "split_name": split_name}},
        {
            "type": "intersection",
            "intersection": {
                "split_name": split_name,
                "pixel_id": 0,
                "gaussian_id": 0,
                "opacity": 0.2,
                "depth": 8.0,
                "xyz": [0.0, 0.0, 8.0],
            },
        },
        {
            "type": "intersection",
            "intersection": {
                "split_name": split_name,
                "pixel_id": 0,
                "gaussian_id": 1,
                "opacity": 0.8,
                "depth": 3.0,
                "xyz": [0.0, 0.0, 3.0],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_raster_ray_contributors_cli_writes_audited_source_rays(tmp_path):
    intersections = tmp_path / "intersections.jsonl"
    output_dir = tmp_path / "source_rays"
    _write_intersections(intersections)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_raster_ray_contributors",
            "--intersections",
            str(intersections),
            "--output_dir",
            str(output_dir),
            "--scene",
            "ToyScene",
            "--source_view_id",
            "s1",
            "--split_name",
            "selfmap_train",
            "--width",
            "10",
            "--top_k",
            "2",
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
    assert rows[1]["ray"]["contributors"][0]["gaussian_id"] == 1

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert manifest["scene"] == "ToyScene"
    assert metrics["ray_count"] == 1
    assert split_audit["audit_status"] == "passed"
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_raster_ray_contributors_cli_rejects_test_split(tmp_path):
    intersections = tmp_path / "intersections.jsonl"
    _write_intersections(intersections, split_name="test")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_raster_ray_contributors",
            "--intersections",
            str(intersections),
            "--output_dir",
            str(tmp_path / "source_rays"),
            "--scene",
            "ToyScene",
            "--source_view_id",
            "s1",
            "--split_name",
            "test",
            "--width",
            "10",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "test split" in completed.stderr
