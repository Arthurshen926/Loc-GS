import numpy as np
import pickle
import pytest
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from loc_gs.stdloc_native.unified_lff_export import build_unified_lff_map


def _write_ply(path):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("loc_0", "f4"), ("loc_1", "f4")]
    data = np.empty(3, dtype=dtype)
    data["x"] = [0.0, 1.0, 2.0]
    data["y"] = [0.0, 1.0, 2.0]
    data["z"] = [1.0, 1.0, 1.0]
    data["loc_0"] = [1.0, 0.0, 1.0]
    data["loc_1"] = [0.0, 2.0, 1.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def _write_ply_with_locability(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("locability_logit", "f4"),
        ("loc_0", "f4"),
        ("loc_1", "f4"),
    ]
    data = np.empty(3, dtype=dtype)
    data["x"] = [0.0, 1.0, 2.0]
    data["y"] = [0.0, 1.0, 2.0]
    data["z"] = [1.0, 1.0, 1.0]
    data["locability_logit"] = [0.0, 0.0, 0.0]
    data["loc_0"] = [1.0, 0.0, 1.0]
    data["loc_1"] = [0.0, 2.0, 1.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def _write_geometry_only_ply(path):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    data = np.empty(3, dtype=dtype)
    data["x"] = [0.0, 1.0, 2.0]
    data["y"] = [0.0, 1.0, 2.0]
    data["z"] = [1.0, 1.0, 1.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def test_build_unified_lff_map_updates_only_mapped_gaussian_descriptors(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    pc_dir.mkdir(parents=True)
    _write_ply(pc_dir / "point_cloud.ply")
    _write_ply(source / "input.ply")
    (source / "detector").mkdir()
    (source / "detector" / "marker.txt").write_text("keep", encoding="utf-8")
    checkpoint = tmp_path / "unified.pt"
    export_desc = F.normalize(torch.tensor([[1.0, 1.0]], dtype=torch.float32), p=2, dim=-1)
    torch.save(
        {
            "config": {"model_type": "unified_lff_descriptor"},
            "export_descriptors": export_desc,
            "base_gaussian_id": torch.tensor([1], dtype=torch.long),
        },
        checkpoint,
    )
    output = tmp_path / "output"

    manifest = build_unified_lff_map(
        source_map=source,
        output_map=output,
        checkpoint_path=checkpoint,
    )

    vertex = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    assert np.allclose(vertex["loc_0"][0], 1.0)
    assert np.allclose(vertex["loc_1"][0], 0.0)
    assert np.allclose([vertex["loc_0"][1], vertex["loc_1"][1]], [np.sqrt(2.0), np.sqrt(2.0)], atol=1e-6)
    assert (output / "detector" / "marker.txt").exists()
    assert manifest["updated_gaussians"] == 1
    assert manifest["single_path_deployment"] is True


def test_build_unified_lff_map_skips_root_input_without_descriptor_fields(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    pc_dir.mkdir(parents=True)
    _write_ply(pc_dir / "point_cloud.ply")
    _write_geometry_only_ply(source / "input.ply")
    checkpoint = tmp_path / "unified.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.tensor([[1.0, 1.0]], dtype=torch.float32), p=2, dim=-1),
            "base_gaussian_id": torch.tensor([1], dtype=torch.long),
        },
        checkpoint,
    )

    manifest = build_unified_lff_map(
        source_map=source,
        output_map=tmp_path / "output",
        checkpoint_path=checkpoint,
    )

    assert manifest["root_input_updated"] is False


def test_build_unified_lff_map_can_export_gate_to_locability_and_detector_scores(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply_with_locability(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long), handle)
    with (detector / "sampled_scores.pkl").open("wb") as handle:
        pickle.dump(
            {
                "sampled_scores": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                "score_avg": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            },
            handle,
        )
    checkpoint = tmp_path / "unified.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.tensor([[1.0, 1.0]], dtype=torch.float32), p=2, dim=-1),
            "base_gaussian_id": torch.tensor([1], dtype=torch.long),
            "gate": torch.tensor([0.9], dtype=torch.float32),
        },
        checkpoint,
    )
    output = tmp_path / "output"

    build_unified_lff_map(
        source_map=source,
        output_map=output,
        checkpoint_path=checkpoint,
        gate_locability_blend=1.0,
    )

    vertex = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    assert np.isclose(1.0 / (1.0 + np.exp(-vertex["locability_logit"][1])), 0.9, atol=1e-4)
    with (output / "detector" / "sampled_scores.pkl").open("rb") as handle:
        scores = pickle.load(handle)
    assert torch.allclose(scores["sampled_scores"], torch.tensor([0.1, 0.9, 0.3]))
    assert torch.allclose(scores["score_avg"], torch.tensor([0.1, 0.9, 0.3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor pickle regression needs CUDA")
def test_build_unified_lff_map_accepts_cuda_detector_score_pickles(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply_with_locability(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2], dtype=torch.long, device="cuda"), handle)
    with (detector / "sampled_scores.pkl").open("wb") as handle:
        pickle.dump(
            {
                "sampled_scores": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device="cuda"),
                "score_avg": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device="cuda"),
            },
            handle,
        )
    checkpoint = tmp_path / "unified.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(torch.tensor([[1.0, 1.0]], dtype=torch.float32), p=2, dim=-1),
            "base_gaussian_id": torch.tensor([1], dtype=torch.long),
            "gate": torch.tensor([0.9], dtype=torch.float32),
        },
        checkpoint,
    )

    build_unified_lff_map(
        source_map=source,
        output_map=tmp_path / "output",
        checkpoint_path=checkpoint,
        gate_locability_blend=1.0,
    )

    with (tmp_path / "output" / "detector" / "sampled_scores.pkl").open("rb") as handle:
        scores = pickle.load(handle)
    assert scores["sampled_scores"].device.type == "cpu"
    assert torch.allclose(scores["sampled_scores"], torch.tensor([0.1, 0.9, 0.3]))
