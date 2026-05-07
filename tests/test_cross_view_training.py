import torch
import torch.nn.functional as F

from loc_gs.losses.cross_view import (
    DescriptorMemoryBank,
    cross_view_projective_contrastive_loss,
    projective_view_overlap,
)


def test_projective_view_overlap_counts_shared_visible_points():
    points = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [10.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    pose = torch.eye(4)
    K = torch.tensor([[4.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])

    overlap = projective_view_overlap(points, pose, pose, K, height=4, width=4)

    assert 0.0 < overlap <= 1.0


def test_descriptor_memory_bank_updates_valid_rows_with_ema():
    bank = DescriptorMemoryBank(num_embeddings=4, dim=3, momentum=0.5)
    desc = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dim=-1)

    bank.update(torch.tensor([1, 3]), desc)

    gathered = bank.lookup(torch.tensor([0, 1, 3]))
    assert torch.allclose(gathered[0], torch.zeros(3))
    assert torch.allclose(gathered[1], desc[0])
    assert torch.allclose(gathered[2], desc[1])
    assert bank.valid_mask.tolist() == [False, True, False, True]


def test_cross_view_projective_contrastive_loss_backprops_to_descriptors():
    desc_a = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_b = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_a.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_b.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_b.data[:, :, 1, 1] = torch.tensor([0.0, 1.0])
    desc_a = F.normalize(desc_a, dim=1)
    desc_b = F.normalize(desc_b, dim=1)
    depth = torch.ones(1, 2, 2)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)

    out = cross_view_projective_contrastive_loss(
        desc_a,
        depth,
        pose,
        desc_b,
        pose,
        K,
        max_samples=4,
        temperature=0.1,
        hard_negative_weight=0.5,
    )

    assert torch.isfinite(out["total"])
    assert out["valid_samples"].item() > 0
    out["total"].backward()
    assert desc_a.grad is not None
    assert desc_b.grad is not None


def test_cross_view_projective_contrastive_loss_filters_depth_inconsistent_targets():
    desc_a = F.normalize(torch.randn(1, 4, 3, 3), dim=1)
    desc_b = F.normalize(torch.randn(1, 4, 3, 3), dim=1)
    depth_a = torch.ones(1, 3, 3)
    depth_b = torch.full((1, 3, 3), 2.0)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)

    out = cross_view_projective_contrastive_loss(
        desc_a,
        depth_a,
        pose,
        desc_b,
        pose,
        K,
        depth_b=depth_b,
        max_samples=9,
        depth_tolerance=0.01,
        depth_rel_tolerance=0.0,
    )

    assert out["valid_samples"].item() == 0
    assert torch.allclose(out["total"], torch.tensor(0.0))


def test_cross_view_projective_contrastive_loss_can_use_teacher_projected_positive():
    desc_a_good = torch.zeros(1, 2, 2, 2, requires_grad=True)
    desc_a_bad = torch.zeros(1, 2, 2, 2, requires_grad=True)
    model_desc_b = torch.randn(1, 2, 2, 2)
    teacher_desc_b = torch.zeros(1, 2, 2, 2)
    valid_a = torch.zeros(1, 2, 2, dtype=torch.bool)
    valid_a[:, 0, 0] = True

    desc_a_good.data[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    desc_a_bad.data[:, :, 0, 0] = torch.tensor([0.0, 1.0])
    teacher_desc_b[:, :, 0, 0] = torch.tensor([1.0, 0.0])
    teacher_desc_b[:, :, 0, 1] = torch.tensor([0.0, 1.0])
    teacher_desc_b[:, :, 1, 0] = torch.tensor([0.0, 1.0])
    teacher_desc_b[:, :, 1, 1] = torch.tensor([0.0, 1.0])

    depth = torch.ones(1, 2, 2)
    pose = torch.eye(4).unsqueeze(0)
    K = torch.eye(3)

    common = dict(
        depth_a=depth,
        pose_a_w2c=pose,
        desc_b=model_desc_b,
        pose_b_w2c=pose,
        K=K,
        valid_a=valid_a,
        positive_desc_b=teacher_desc_b,
        max_samples=1,
        temperature=0.1,
        hard_negative_weight=0.5,
        hard_negative_exclusion_radius=0.0,
    )
    good = cross_view_projective_contrastive_loss(desc_a=desc_a_good, **common)
    bad = cross_view_projective_contrastive_loss(desc_a=desc_a_bad, **common)

    assert good["total"] < bad["total"]
    good["total"].backward()
    assert desc_a_good.grad is not None
    assert desc_a_good.grad.abs().sum() > 0
