import json
import struct
from pathlib import Path

import torch

from loc_gs.scripts.run_internal_sparse_smoke import main


def _write_ply(path: Path) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 6\n"
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
    ]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<3f", *row))
    return path


def _write_artifact(path: Path) -> Path:
    payload = {
        "metadata": {
            "format": "listwise",
            "scene": "GreatCourt",
            "source_split_name": "train",
            "topk": 1,
            "split_audit": {"audit_status": "passed", "checks": {}},
        },
        "base_gaussian_id": torch.arange(6, dtype=torch.int64),
        "query_yx": torch.tensor(
            [
                [71.3333, 96.6667],
                [71.9355, 142.5806],
                [114.1379, 100.6897],
                [111.2121, 136.9697],
                [90.0, 120.0],
                [94.1176, 91.1765],
            ],
            dtype=torch.float32,
        ),
        "landmark_id": torch.arange(6, dtype=torch.int64).reshape(6, 1),
        "cosine": torch.ones((6, 1), dtype=torch.float32),
        "label": torch.zeros(6, dtype=torch.int64),
        "candidate_mask": torch.ones((6, 1), dtype=torch.bool),
        "query_id": [f"img.png::kp{i}" for i in range(6)],
        "image_id": ["img.png"] * 6,
        "keypoint_id": [f"kp{i}" for i in range(6)],
        "source_phase": ["train"] * 6,
    }
    torch.save(payload, path)
    return path


def test_internal_sparse_smoke_cli_runs_cached_candidates_through_pnp(tmp_path: Path):
    cameras = tmp_path / "cameras.json"
    cameras.write_text(
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
    out = tmp_path / "smoke"

    rc = main(
        [
            "--scene",
            "GreatCourt",
            "--split_name",
            "train",
            "--candidate_artifact",
            str(_write_artifact(tmp_path / "pairs.pt")),
            "--point_cloud",
            str(_write_ply(tmp_path / "point_cloud.ply")),
            "--cameras_json",
            str(cameras),
            "--output_dir",
            str(out),
            "--max_queries",
            "1",
            "--score_mode",
            "native",
        ]
    )

    assert rc == 0
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert metrics["success_count"] == 1
    assert metrics["mean_inliers"] >= 4
    assert metrics["median_te_cm"] < 1.0
    assert metrics["median_re_deg"] < 1.0
    assert metrics["pose_metric_status"] == "computed_unverified"
    assert manifest["inference_stage"] == "sparse_only"
    assert manifest["hyperparameters"]["score_mode"] == "native"
