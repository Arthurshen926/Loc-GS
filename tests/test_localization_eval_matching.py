import torch

from loc_gs.scripts.eval_localization import _subpixel_refine_matches


def test_subpixel_refine_matches_returns_softargmax_position():
    sim = torch.full((1, 25), -5.0)
    sim[0, 12] = 1.0  # y=2, x=2
    sim[0, 13] = 0.8  # y=2, x=3 pulls the local expectation right

    pos = _subpixel_refine_matches(
        sim=sim,
        query_indices=torch.tensor([0]),
        best_indices=torch.tensor([12]),
        height=5,
        width=5,
        window_radius=1,
        temperature=0.5,
    )

    assert pos.shape == (1, 2)
    assert 2.0 < pos[0, 1] < 2.5
    assert torch.isclose(pos[0, 0], torch.tensor(2.0), atol=0.05)
