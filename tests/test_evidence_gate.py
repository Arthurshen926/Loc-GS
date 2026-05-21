import torch

from loc_gs.stdloc_native.evidence_gate import build_evidence_gate


def test_evidence_gate_rejects_high_selector_without_positive_or_alpha_evidence():
    gate = build_evidence_gate(
        selector=torch.tensor([0.95, 0.80, 0.70], dtype=torch.float32),
        positive_support=torch.tensor([0.0, 0.8, 0.8], dtype=torch.float32),
        hard_negative_risk=torch.tensor([0.0, 0.2, 0.9], dtype=torch.float32),
        alpha_reliability=torch.tensor([1.0, 0.7, 0.7], dtype=torch.float32),
        pose_utility=torch.tensor([1.0, 0.4, 0.4], dtype=torch.float32),
        min_positive_support=0.1,
        max_hard_negative_risk=0.5,
        min_alpha_reliability=0.5,
        min_pose_utility=0.1,
    )

    assert gate["mask"].tolist() == [False, True, False]
    assert int(gate["score"].argmax().item()) == 1
    assert gate["metadata"]["accepted_count"] == 1
    assert gate["metadata"]["rejected_by_positive_support"] == 1
    assert gate["metadata"]["rejected_by_hard_negative"] == 1


def test_evidence_gate_can_keep_native_source_even_without_extra_evidence():
    gate = build_evidence_gate(
        positive_support=torch.tensor([0.0, 0.8], dtype=torch.float32),
        hard_negative_risk=torch.tensor([0.9, 0.0], dtype=torch.float32),
        source_idx=torch.tensor([0], dtype=torch.long),
        keep_source=True,
        min_positive_support=0.1,
        max_hard_negative_risk=0.5,
    )

    assert gate["mask"].tolist() == [True, True]
    assert gate["metadata"]["source_forced_count"] == 1

