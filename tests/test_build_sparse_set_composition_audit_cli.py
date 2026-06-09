import json
import pickle

import torch

from loc_gs.scripts.build_sparse_set_composition_audit import main


def test_build_sparse_set_composition_audit_cli_writes_report_bundle(tmp_path):
    selected = tmp_path / "selected.pkl"
    baseline = tmp_path / "baseline.pkl"
    profile = tmp_path / "profile.json"
    output = tmp_path / "audit"
    with selected.open("wb") as handle:
        pickle.dump(torch.tensor([1, 2]), handle)
    with baseline.open("wb") as handle:
        pickle.dump(torch.tensor([0, 1]), handle)
    profile.write_text(
        json.dumps(
            {
                "schema": "loc_gs_sparse_pnp_validation_profile_v2",
                "split_name": "train_dev",
                "attribution_status": "attributed",
                "protected_per_query_support": {"q": {"0": 1.0, "1": 1.0}},
                "validated_per_query_support": {},
                "landmark_regression_risk": {"2": 3.0},
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--selected_idx",
            str(selected),
            "--baseline_idx",
            str(baseline),
            "--sparse_validation_profile",
            str(profile),
            "--output_dir",
            str(output),
            "--scene",
            "ShopFacade",
        ]
    )

    assert rc == 0
    report = json.loads((output / "report.json").read_text())
    assert report["baseline_overlap_count"] == 1
    assert report["protected_support"]["support_mass_selected_fraction"] == 0.5
    assert (output / "report.md").exists()
    assert (output / "manifest.json").exists()
    assert (output / "split_audit.json").exists()

