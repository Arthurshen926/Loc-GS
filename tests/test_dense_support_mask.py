import torch

from loc_gs.dense_support.rendered_support_mask import build_dense_residual_support


def test_build_dense_residual_support_combines_lsf_and_pose_leverage():
    payload = build_dense_residual_support(
        contributor_ids=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        contribution_weights=torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=torch.float32),
        gaussian_support=torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        pose_leverage=torch.tensor([1.0, 0.1], dtype=torch.float32),
        threshold=0.3,
    )

    assert payload["residual_support"][0] > payload["residual_support"][1]
    assert payload["keep"].tolist() == [True, False]
    assert payload["metadata"]["dense_support_version"] == "v1_proxy"

