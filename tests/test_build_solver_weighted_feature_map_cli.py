import json
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def _write_tiny_map(root):
    pc_dir = root / "point_cloud" / "iteration_30000"
    detector = root / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir(parents=True)
    (root / "cfg_args").write_text("cfg", encoding="utf-8")
    (detector / "sampled_idx.pkl").write_bytes(b"idx")
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("loc_0", "f4"), ("loc_1", "f4")]
    data = np.empty(3, dtype=dtype)
    data["x"] = [0.0, 1.0, 2.0]
    data["y"] = [0.0, 1.0, 2.0]
    data["z"] = [0.0, 1.0, 2.0]
    data["loc_0"] = [3.0, 0.0, 1.0]
    data["loc_1"] = [4.0, 2.0, 1.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(pc_dir / "point_cloud.ply")


def test_build_solver_weighted_feature_map_cli_materializes_full_ply(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    artifact = tmp_path / "selected_landmark_descriptors.pt"
    _write_tiny_map(source)
    torch.save(
        {
            "descriptors": F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=-1),
            "base_descriptors": F.normalize(torch.tensor([[0.0, 2.0]], dtype=torch.float32), p=2, dim=-1),
            "gaussian_ids": torch.tensor([1], dtype=torch.long),
            "metadata": {"descriptor_mode": "solver_weighted_pair_cache_fusion", "positive_pair_count": 1},
        },
        artifact,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_weighted_feature_map",
            "--source_map",
            str(source),
            "--descriptor_artifact",
            str(artifact),
            "--output_map",
            str(output),
            "--scene",
            "ToyScene",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    vertex = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    assert np.allclose([vertex["loc_0"][0], vertex["loc_1"][0]], [3.0, 4.0])
    assert np.allclose([vertex["loc_0"][1], vertex["loc_1"][1]], [2.0, 0.0], atol=1e-6)
    manifest = json.loads((output / "solver_weighted_feature_map_manifest.json").read_text())
    assert manifest["scene"] == "ToyScene"
    assert manifest["updated_gaussian_count"] == 1
    assert manifest["single_path_deployment"] is True


def test_build_solver_weighted_feature_map_cli_rejects_incompatible_artifact_source(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    artifact = tmp_path / "selected_landmark_descriptors.pt"
    _write_tiny_map(source)
    torch.save(
        {
            "descriptors": F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=-1),
            "base_descriptors": F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=-1),
            "gaussian_ids": torch.tensor([1], dtype=torch.long),
            "metadata": {"descriptor_mode": "solver_weighted_pair_cache_fusion", "positive_pair_count": 1},
        },
        artifact,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.build_solver_weighted_feature_map",
            "--source_map",
            str(source),
            "--descriptor_artifact",
            str(artifact),
            "--output_map",
            str(output),
            "--scene",
            "ToyScene",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "descriptor artifact is not compatible" in result.stderr
