import pickle

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from loc_gs.stdloc_native.unified_lff_export import build_unified_lff_map


def _write_ply_with_locability(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("locability_logit", "f4"),
        ("loc_0", "f4"),
        ("loc_1", "f4"),
    ]
    data = np.empty(4, dtype=dtype)
    data["x"] = [0.0, 1.0, 2.0, 3.0]
    data["y"] = [0.0, 1.0, 2.0, 3.0]
    data["z"] = [1.0, 1.0, 1.0, 1.0]
    data["locability_logit"] = [0.0, 0.0, 0.0, 0.0]
    data["loc_0"] = [1.0, 0.0, 1.0, 2.0]
    data["loc_1"] = [0.0, 2.0, 1.0, 0.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def _make_source(tmp_path):
    source = tmp_path / "source"
    pc_dir = source / "point_cloud" / "iteration_30000"
    detector = source / "detector"
    pc_dir.mkdir(parents=True)
    detector.mkdir()
    _write_ply_with_locability(pc_dir / "point_cloud.ply")
    with (detector / "sampled_idx.pkl").open("wb") as handle:
        pickle.dump(torch.tensor([0, 1, 2, 3], dtype=torch.long), handle)
    with (detector / "sampled_scores.pkl").open("wb") as handle:
        pickle.dump(
            {
                "sampled_scores": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
                "score_avg": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            },
            handle,
        )
    checkpoint = tmp_path / "unified.pt"
    torch.save(
        {
            "export_descriptors": F.normalize(
                torch.tensor([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [6.0, 6.0]], dtype=torch.float32),
                p=2,
                dim=-1,
            ),
            "base_gaussian_id": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "gate": torch.tensor([0.2, 0.4, 0.7, 0.9], dtype=torch.float32),
        },
        checkpoint,
    )
    return source, checkpoint


def _read_locability_probs(output):
    vertex = PlyData.read(str(output / "point_cloud" / "iteration_30000" / "point_cloud.ply"))["vertex"].data
    logits = np.asarray(vertex["locability_logit"], dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def _read_sampled_scores(output):
    with (output / "detector" / "sampled_scores.pkl").open("rb") as handle:
        return pickle.load(handle)["sampled_scores"].float()


def test_uniform_selector_ablation_uses_constant_gate_mean(tmp_path):
    source, checkpoint = _make_source(tmp_path)
    output = tmp_path / "uniform"

    manifest = build_unified_lff_map(
        source_map=source,
        output_map=output,
        checkpoint_path=checkpoint,
        descriptor_mode="native",
        gate_locability_blend=1.0,
        gate_transform="uniform",
        ablation_type="uniform",
    )

    expected = torch.full((4,), 0.55)
    assert np.allclose(_read_locability_probs(output), expected.numpy(), atol=1e-4)
    assert torch.allclose(_read_sampled_scores(output), expected, atol=1e-6)
    assert manifest["ablation_type"] == "uniform"
    assert manifest["gate_transform"] == "uniform"


def test_permuted_selector_ablation_is_seeded_and_preserves_values(tmp_path):
    source, checkpoint = _make_source(tmp_path)
    out_a = tmp_path / "perm_a"
    out_b = tmp_path / "perm_b"

    build_unified_lff_map(
        source_map=source,
        output_map=out_a,
        checkpoint_path=checkpoint,
        descriptor_mode="native",
        gate_locability_blend=1.0,
        gate_transform="permuted",
        gate_transform_seed=7,
        ablation_type="permuted",
    )
    build_unified_lff_map(
        source_map=source,
        output_map=out_b,
        checkpoint_path=checkpoint,
        descriptor_mode="native",
        gate_locability_blend=1.0,
        gate_transform="permuted",
        gate_transform_seed=7,
        ablation_type="permuted",
    )

    scores_a = _read_sampled_scores(out_a)
    scores_b = _read_sampled_scores(out_b)
    assert torch.allclose(scores_a, scores_b)
    assert torch.allclose(torch.tensor(sorted(float(v) for v in scores_a)), torch.tensor([0.2, 0.4, 0.7, 0.9]))
    assert not torch.allclose(scores_a, torch.tensor([0.2, 0.4, 0.7, 0.9]))


def test_inverted_selector_ablation_and_apply_toggles(tmp_path):
    source, checkpoint = _make_source(tmp_path)
    detector_only = tmp_path / "detector_only"
    locability_only = tmp_path / "locability_only"

    manifest_detector = build_unified_lff_map(
        source_map=source,
        output_map=detector_only,
        checkpoint_path=checkpoint,
        descriptor_mode="native",
        gate_locability_blend=1.0,
        gate_transform="inverted",
        apply_to_detector_scores=True,
        apply_to_ply_locability=False,
        ablation_type="detector_only",
    )
    manifest_locability = build_unified_lff_map(
        source_map=source,
        output_map=locability_only,
        checkpoint_path=checkpoint,
        descriptor_mode="native",
        gate_locability_blend=1.0,
        gate_transform="inverted",
        apply_to_detector_scores=False,
        apply_to_ply_locability=True,
        ablation_type="locability_only",
    )

    inverted = torch.tensor([0.8, 0.6, 0.3, 0.1])
    assert np.allclose(_read_locability_probs(detector_only), np.full((4,), 0.5), atol=1e-4)
    assert torch.allclose(_read_sampled_scores(detector_only), inverted, atol=1e-6)
    assert np.allclose(_read_locability_probs(locability_only), inverted.numpy(), atol=1e-4)
    assert torch.allclose(_read_sampled_scores(locability_only), torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert manifest_detector["apply_to_detector_scores"] is True
    assert manifest_detector["apply_to_ply_locability"] is False
    assert manifest_locability["apply_to_detector_scores"] is False
    assert manifest_locability["apply_to_ply_locability"] is True
