import json
import pickle
import subprocess
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _dump_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _write_tiny_map(root):
    pc_dir = root / "point_cloud" / "iteration_30000"
    pc_dir.mkdir(parents=True)
    data = np.empty(
        3,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    data["x"] = [0.0, 1.0, 2.0]
    data["y"] = [0.0, 1.0, 2.0]
    data["z"] = [0.0, 1.0, 2.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(pc_dir / "point_cloud.ply")
    _dump_pickle(root / "detector" / "sampled_idx.pkl", torch.tensor([0, 1], dtype=torch.long))
    _dump_pickle(root / "detector" / "sampled_scores.pkl", torch.ones(2, dtype=torch.float32))


def test_export_dense_lsf_locability_map_writes_fixed_dense_prior_and_audit_bundle(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    target_path = tmp_path / "dense_lsf_targets.pt"
    _write_tiny_map(source)
    torch.save(
        {
            "dense_lsf_target": torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
            "metadata": {
                "scene": "ToyScene",
                "split_name": "selfmap_train",
                "usage_scope": "dense_residual_teacher_only",
                "sparse_selector_safe": False,
                "split_audit": {"audit_status": "passed", "split_name": "selfmap_train"},
            },
        },
        target_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_dense_lsf_locability_map",
            "--source_map",
            str(source),
            "--dense_lsf_target_path",
            str(target_path),
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
    data = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))
    assert "locability_logit" in data.dtype.names
    assert data["locability_logit"][1] > data["locability_logit"][0]
    assert pickle.load((output / "detector" / "sampled_idx.pkl").open("rb")).tolist() == [0, 1]
    assert manifest["method"] == "loc_gs_dense_lsf_fixed_locability_prior"
    assert manifest["same_budget"] is True
    assert manifest["single_path_deployment"] is True
    assert manifest["branch_selection"] is False
    assert manifest["dense_support"]["source"] == "dense_lsf_target"
    assert split_audit["audit_status"] == "passed"
    assert (output / "command.txt").exists()
    assert (output / "metrics_summary.json").exists()
    assert (output / "git_status.txt").exists()


def test_export_dense_lsf_locability_map_rejects_test_split_artifact(tmp_path):
    source = tmp_path / "source_map"
    output = tmp_path / "output_map"
    target_path = tmp_path / "dense_lsf_targets.pt"
    _write_tiny_map(source)
    torch.save(
        {
            "dense_lsf_target": torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32),
            "metadata": {"scene": "ToyScene", "split_name": "test"},
        },
        target_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.export_dense_lsf_locability_map",
            "--source_map",
            str(source),
            "--dense_lsf_target_path",
            str(target_path),
            "--output_map",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split dense LSF targets are not allowed" in result.stderr
