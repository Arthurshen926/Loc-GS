import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from loc_gs.stdloc_native import lff_export


def _write_tiny_ply(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("locability_logit", "f4"),
        ("loc_0", "f4"),
        ("loc_1", "f4"),
    ]
    data = np.empty(3, dtype=dtype)
    data["x"] = [1.0, 2.0, 3.0]
    data["y"] = [4.0, 5.0, 6.0]
    data["z"] = [7.0, 8.0, 9.0]
    data["locability_logit"] = [-1.0, 0.0, 1.0]
    data["loc_0"] = [1.0, 0.0, 1.0]
    data["loc_1"] = [0.0, 1.0, 1.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def _write_tiny_ply_without_locability(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("loc_0", "f4"),
        ("loc_1", "f4"),
    ]
    data = np.empty(2, dtype=dtype)
    data["x"] = [1.0, 2.0]
    data["y"] = [3.0, 4.0]
    data["z"] = [5.0, 6.0]
    data["loc_0"] = [1.0, 0.0]
    data["loc_1"] = [0.0, 1.0]
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def test_protected_lff_descriptor_export_blends_residual_by_gate_and_reliability():
    ply = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=-1)
    hybrid = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), p=2, dim=-1)
    gate = torch.tensor([1.0, 0.25])

    exported = lff_export.protected_lff_descriptors(
        ply,
        hybrid,
        gate=gate,
        alpha_max=0.2,
        reliability=0.5,
    )

    expected = F.normalize(
        ply + 0.1 * gate[:, None] * (hybrid - ply),
        p=2,
        dim=-1,
    )
    assert torch.allclose(exported, expected)


def test_restore_descriptor_norms_preserves_source_ply_magnitude():
    exported_direction = F.normalize(torch.tensor([[0.6, 0.8], [1.0, 1.0]]), p=2, dim=-1)
    source = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    restored = lff_export.restore_descriptor_norms(exported_direction, source)

    assert torch.allclose(restored.norm(dim=-1), source.norm(dim=-1))


def test_unified_selector_combines_locability_and_sampled_matchability():
    locability = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
    sampled_idx = torch.tensor([1, 3], dtype=torch.long)
    calibrated = torch.tensor([0.9, 0.1], dtype=torch.float32)

    selector, stats = lff_export.build_unified_selector(
        locability=locability,
        calibrated_matchability=calibrated,
        sampled_idx=sampled_idx,
        mode="combined",
        matchability_weight=0.5,
    )

    assert torch.allclose(selector, torch.tensor([0.2, 0.7, 0.6, 0.4]))
    assert stats["selector_mode"] == "combined"
    assert stats["matchability_available"] == 1.0


def test_unified_selector_boosts_matchability_without_suppressing_locability():
    locability = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32)
    sampled_idx = torch.tensor([1, 3], dtype=torch.long)
    calibrated = torch.tensor([0.9, 0.1], dtype=torch.float32)

    selector, stats = lff_export.build_unified_selector(
        locability=locability,
        calibrated_matchability=calibrated,
        sampled_idx=sampled_idx,
        mode="reliability_boost",
        matchability_weight=1.0,
    )

    assert torch.allclose(selector, torch.tensor([0.2, 1.0, 0.6, 0.8]))
    assert stats["selector_mode"] == "reliability_boost"
    assert stats["matchability_available"] == 1.0


def test_unified_selector_applies_floor_and_power():
    selector, stats = lff_export.build_unified_selector(
        locability=torch.tensor([0.25, 1.0], dtype=torch.float32),
        mode="locability",
        floor=0.2,
        power=2.0,
    )

    assert torch.allclose(selector, torch.tensor([0.25, 1.0]))
    assert stats["selector_floor"] == 0.2
    assert stats["selector_power"] == 2.0


def test_blend_locability_logits_can_boost_with_unified_selector():
    source_logits = torch.logit(torch.tensor([0.2, 0.7], dtype=torch.float32))
    lff_logits = torch.logit(torch.tensor([0.4, 0.6], dtype=torch.float32))
    selector = torch.tensor([0.9, 0.1], dtype=torch.float32)

    logits = lff_export._blend_locability_logits(
        source_logits,
        lff_logits,
        reliability=0.5,
        mode="boost",
        selector=selector,
        selector_weight=0.5,
    )

    assert torch.allclose(torch.sigmoid(logits), torch.tensor([0.6, 0.7]), atol=1e-6)


def test_write_lff_point_cloud_replaces_loc_fields_and_locability(tmp_path):
    source = tmp_path / "source.ply"
    output = tmp_path / "output.ply"
    _write_tiny_ply(source)
    descriptors = torch.tensor(
        [
            [0.5, 0.5],
            [0.2, 0.8],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

    lff_export.write_lff_point_cloud(
        source_ply=source,
        output_ply=output,
        descriptors=descriptors,
        locability_logits=logits,
    )

    vertex = PlyData.read(str(output))["vertex"].data
    assert np.allclose(vertex["x"], [1.0, 2.0, 3.0])
    assert np.allclose(vertex["y"], [4.0, 5.0, 6.0])
    assert np.allclose(vertex["loc_0"], [0.5, 0.2, 0.9])
    assert np.allclose(vertex["loc_1"], [0.5, 0.8, 0.1])
    assert np.allclose(vertex["locability_logit"], [2.0, 3.0, 4.0])


def test_write_lff_point_cloud_rejects_descriptor_shape_mismatch(tmp_path):
    source = tmp_path / "source.ply"
    output = tmp_path / "output.ply"
    _write_tiny_ply(source)

    try:
        lff_export.write_lff_point_cloud(
            source_ply=source,
            output_ply=output,
            descriptors=torch.ones(3, 3),
        )
    except ValueError as exc:
        assert "descriptor shape" in str(exc)
    else:
        raise AssertionError("expected descriptor shape mismatch to fail")


def test_write_lff_point_cloud_adds_missing_locability_field(tmp_path):
    source = tmp_path / "source_no_locability.ply"
    output = tmp_path / "output.ply"
    _write_tiny_ply_without_locability(source)

    lff_export.write_lff_point_cloud(
        source_ply=source,
        output_ply=output,
        descriptors=torch.eye(2),
        locability_logits=torch.tensor([0.25, 0.75]),
    )

    vertex = PlyData.read(str(output))["vertex"].data
    assert "locability_logit" in vertex.dtype.names
    assert np.allclose(vertex["locability_logit"], [0.25, 0.75])
