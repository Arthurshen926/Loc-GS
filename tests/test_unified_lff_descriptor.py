import torch
import torch.nn.functional as F

from loc_gs.models.unified_lff import UnifiedLFFDescriptor


def test_unified_lff_descriptor_starts_as_frozen_stdloc_bank():
    base = torch.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)

    model = UnifiedLFFDescriptor(base, alpha_max=0.05, init_gate=0.25)
    exported = model()

    assert torch.allclose(exported, F.normalize(base, p=2, dim=-1))
    assert model.base_descriptors.requires_grad is False
    assert model.residual.requires_grad is True
    assert model.gate_logit.requires_grad is True


def test_unified_lff_descriptor_can_learn_to_suppress_or_apply_residuals():
    base = F.normalize(torch.eye(3, dtype=torch.float32), p=2, dim=-1)
    residual = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    model = UnifiedLFFDescriptor(base, residual_init=residual, alpha_max=0.5, init_gate=0.5)
    with torch.no_grad():
        model.gate_logit[0] = -30.0
        model.gate_logit[1] = 30.0

    exported = model(torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(exported[0], base[0], atol=1e-6)
    expected_applied = F.normalize(base[1] + 0.5 * residual[1], p=2, dim=-1)
    assert torch.allclose(exported[1], expected_applied, atol=1e-6)


def test_unified_lff_selector_is_independent_from_descriptor_residual_gate():
    base = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)

    model = UnifiedLFFDescriptor(base, alpha_max=0.5, init_gate=0.1, init_selector=0.8)
    with torch.no_grad():
        model.gate_logit.fill_(-30.0)

    exported = model()

    assert torch.allclose(exported, base, atol=1e-6)
    assert torch.allclose(model.gate(), torch.zeros(2), atol=1e-6)
    assert torch.allclose(model.selector(), torch.full((2,), 0.8), atol=1e-6)


def test_unified_lff_trust_region_loss_penalizes_gate_weighted_residual():
    base = F.normalize(torch.eye(2, dtype=torch.float32), p=2, dim=-1)
    residual = torch.ones_like(base)
    model = UnifiedLFFDescriptor(base, residual_init=residual, alpha_max=0.1, init_gate=0.5)

    losses = model.trust_region_loss(l1_weight=0.25)
    losses["loss"].backward()

    assert set(losses) == {"loss", "residual_l2", "alpha_l1"}
    assert losses["loss"].item() > 0.0
    assert model.residual.grad is not None
    assert model.gate_logit.grad is not None


def test_unified_lff_descriptor_alpha_max_bounds_large_raw_residuals():
    base = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), p=2, dim=-1)
    residual = torch.tensor([[0.0, 1000.0]], dtype=torch.float32)
    model = UnifiedLFFDescriptor(base, residual_init=residual, alpha_max=0.05, init_gate=1.0)

    exported = model()

    expected = F.normalize(torch.tensor([[1.0, 0.05]], dtype=torch.float32), p=2, dim=-1)
    assert torch.allclose(exported, expected, atol=1e-6)
