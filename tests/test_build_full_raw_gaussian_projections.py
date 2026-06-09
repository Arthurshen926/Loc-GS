import json
import subprocess
import sys
from pathlib import Path

import torch


def _write_inputs(root: Path, *, split_name: str = "selfmap_train") -> tuple[Path, Path]:
    gaussian_path = root / "gaussians.pt"
    views_path = root / "views.jsonl"
    torch.save(
        {
            "xyz": torch.tensor(
                [
                    [0.0, 0.0, 2.0],
                    [1.0, 0.0, 2.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=torch.float32,
            ),
            "opacity": torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32),
            "radius_px": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        },
        gaussian_path,
    )
    rows = [
        {"type": "manifest", "manifest": {"scene": "ToyScene", "split_name": split_name}},
        {
            "type": "view",
            "view": {
                "source_view_id": "s1",
                "split_name": split_name,
                "width": 100,
                "height": 100,
                "world_to_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsic": [
                    [10.0, 0.0, 50.0],
                    [0.0, 10.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        },
    ]
    views_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return gaussian_path, views_path


def test_build_full_raw_gaussian_projections_cli_writes_audited_projection_bundle(tmp_path):
    gaussian_path, views_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "projections"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_full_raw_gaussian_projections",
            "--gaussians",
            str(gaussian_path),
            "--views",
            str(views_path),
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
    projections_path = output_dir / "projected_gaussians.jsonl"
    assert payload["projections_path"] == str(projections_path)
    rows = [json.loads(line) for line in projections_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "manifest"
    assert rows[0]["manifest"]["projection_source"] == "full_raw_gaussians"
    projection_rows = [row["projection"] for row in rows[1:]]
    assert [row["gaussian_id"] for row in projection_rows] == [0, 1]
    assert all(row["projection_source"] == "full_raw_gaussians" for row in projection_rows)

    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_dir / "split_audit.json").read_text(encoding="utf-8"))
    assert metrics["source_gaussian_count"] == 3
    assert metrics["projected_gaussian_count"] == 2
    assert metrics["sampled_idx_used"] is False
    assert split_audit["audit_status"] == "passed"
    assert (output_dir / "command.txt").exists()
    assert (output_dir / "git_status.txt").exists()


def test_build_full_raw_gaussian_projections_cli_rejects_test_split(tmp_path):
    gaussian_path, views_path = _write_inputs(tmp_path, split_name="test")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_full_raw_gaussian_projections",
            "--gaussians",
            str(gaussian_path),
            "--views",
            str(views_path),
            "--output_dir",
            str(tmp_path / "projections"),
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
