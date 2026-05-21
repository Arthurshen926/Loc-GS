import torch

from loc_gs.diagnostics.rendered_reliability import rendered_support_mask


def test_rendered_support_mask_combines_support_dominance_entropy_and_depth():
    contributor_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=torch.float32)
    support = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    depth_stability = torch.tensor([1.0, 0.2], dtype=torch.float32)

    mask = rendered_support_mask(
        contributor_ids=contributor_ids,
        contribution_weights=weights,
        gaussian_support=support,
        depth_stability=depth_stability,
        threshold=0.3,
    )

    assert mask["support"][0] > mask["support"][1]
    assert mask["reliability"][0] > mask["reliability"][1]
    assert mask["keep"].tolist() == [True, False]
    assert mask["metadata"]["composition_proxy"] is True

