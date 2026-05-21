import torch

from loc_gs.diagnostics.solver_tuple_bank import compare_selection_solver_diagnostics


def _support_count_counterexample_payload():
    query_yx = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    source_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    replacement_ids = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    landmark_id = torch.stack([source_ids, replacement_ids], dim=1)
    return {
        "query_id": torch.zeros(4, dtype=torch.long),
        "query_yx": query_yx,
        "landmark_id": landmark_id,
        "cosine": torch.full((4, 2), 0.9, dtype=torch.float32),
        "candidate_mask": torch.ones((4, 2), dtype=torch.bool),
        "reprojection_error": torch.full((4, 2), 1.0, dtype=torch.float32),
        "metadata": {
            "reprojection_threshold_px": 3.0,
            "split_audit": {"audit_status": "passed"},
        },
    }


def test_tuple_diagnostic_finds_support_count_preserved_but_solver_geometry_lost():
    payload = _support_count_counterexample_payload()
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 5.0],
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [0.0, 0.01, 1.0],
            [0.01, 0.01, 1.0],
        ],
        dtype=torch.float32,
    )

    report = compare_selection_solver_diagnostics(
        payload,
        source_idx=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        candidate_idx=torch.tensor([4, 5, 6, 7], dtype=torch.long),
        num_gaussians=8,
        xyz=xyz,
        tuple_size=4,
        max_tuples_per_query=32,
        min_logdet_h=-5.0,
        min_spread_3d=0.05,
    )

    assert report["summary_delta"]["support_count_sum"] == 0
    assert report["summary_delta"]["viable_tuple_mass"] < 0
    assert report["summary_delta"]["mean_logdet_H"] < 0
    assert report["query_deltas"][0]["support_count_delta"] == 0
    assert report["query_deltas"][0]["mean_logdet_H_delta"] < 0
    assert report["candidate"]["summary"]["viable_tuple_count"] == 0
    assert report["source"]["summary"]["viable_tuple_count"] == 1
