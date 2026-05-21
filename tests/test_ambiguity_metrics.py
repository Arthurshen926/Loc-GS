import torch

from loc_gs.diagnostics.ambiguity_metrics import descriptor_ambiguity_risk


def test_descriptor_ambiguity_risk_increases_for_near_duplicate_descriptors():
    distinct = torch.eye(4, dtype=torch.float32)
    duplicates = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    distinct_risk = descriptor_ambiguity_risk(distinct, cosine_threshold=0.95)
    duplicate_risk = descriptor_ambiguity_risk(duplicates, cosine_threshold=0.95)

    assert distinct_risk["near_duplicate_pairs"] == 0
    assert duplicate_risk["near_duplicate_pairs"] == 1
    assert duplicate_risk["ambiguity_risk"] > distinct_risk["ambiguity_risk"]

