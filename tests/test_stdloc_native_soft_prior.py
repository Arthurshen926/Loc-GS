import pickle
from pathlib import Path

import torch
import yaml

from loc_gs.stdloc_native import soft_prior


def _write_source_map(root: Path) -> None:
    detector = root / "detector"
    detector.mkdir(parents=True)
    pickle.dump(torch.tensor([1, 3], dtype=torch.long), (detector / "sampled_idx.pkl").open("wb"))
    pickle.dump(
        {
            "sampled_scores": torch.tensor([0.2, 0.6], dtype=torch.float32),
            "score_avg": torch.tensor([0.1, 0.2, 0.3, 0.6], dtype=torch.float32),
            "score_num": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        },
        (detector / "sampled_scores.pkl").open("wb"),
    )
    (root / "input.ply").write_text("source-ply\n", encoding="utf-8")


def _write_base_cfg(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "sparse": {
                    "detector_path": "detector/30000_detector.pth",
                    "landmark_path": "detector/sampled_idx.pkl",
                },
                "dense": {"iters": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_rank_normalize_maps_low_to_zero_and_high_to_one():
    ranked = soft_prior.rank_normalize(torch.tensor([0.2, 0.9, 0.4], dtype=torch.float32))

    assert torch.allclose(ranked, torch.tensor([0.0, 1.0, 0.5]))


def test_build_soft_prior_map_with_zero_rho_preserves_scores_and_disables_prior(tmp_path):
    source_map = tmp_path / "source_map"
    output_map = tmp_path / "soft_map"
    calibration_path = tmp_path / "calib.pt"
    base_cfg = tmp_path / "stdloc.yaml"
    _write_source_map(source_map)
    _write_base_cfg(base_cfg)
    torch.save({"landmark_matchability": torch.tensor([0.9, 0.1], dtype=torch.float32)}, calibration_path)

    manifest = soft_prior.build_soft_prior_map(
        source_map=source_map,
        output_map=output_map,
        calibration_path=calibration_path,
        base_cfg_path=base_cfg,
        output_cfg_path=output_map / "stdloc_soft_prior.yaml",
        rho=0.0,
        update_point_cloud_locability=False,
    )

    fused = pickle.load((output_map / "detector/sampled_scores.pkl").open("rb"))
    cfg = yaml.safe_load((output_map / "stdloc_soft_prior.yaml").read_text(encoding="utf-8"))
    assert torch.allclose(fused["sampled_scores"], torch.tensor([0.2, 0.6]))
    assert torch.allclose(fused["score_avg"], torch.tensor([0.1, 0.2, 0.3, 0.6]))
    assert cfg["sparse"]["landmark_prior_weight"] == 0.0
    assert cfg["dense"]["locability_prior_weight"] == 0.0
    assert manifest["rho"] == 0.0


def test_build_soft_prior_map_blends_ranked_selfmap_scores_into_stdloc_map(tmp_path):
    source_map = tmp_path / "source_map"
    output_map = tmp_path / "soft_map"
    calibration_path = tmp_path / "calib.pt"
    base_cfg = tmp_path / "stdloc.yaml"
    _write_source_map(source_map)
    _write_base_cfg(base_cfg)
    torch.save({"landmark_matchability": torch.tensor([0.9, 0.1], dtype=torch.float32)}, calibration_path)

    manifest = soft_prior.build_soft_prior_map(
        source_map=source_map,
        output_map=output_map,
        calibration_path=calibration_path,
        base_cfg_path=base_cfg,
        output_cfg_path=output_map / "stdloc_soft_prior.yaml",
        rho=1.0,
        update_point_cloud_locability=False,
    )

    fused = pickle.load((output_map / "detector/sampled_scores.pkl").open("rb"))
    cfg = yaml.safe_load((output_map / "stdloc_soft_prior.yaml").read_text(encoding="utf-8"))
    assert torch.allclose(fused["sampled_scores"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(fused["score_avg"], torch.tensor([0.1, 1.0, 0.3, 0.0]))
    assert torch.equal(fused["score_num"], torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert cfg["sparse"]["landmark_score_path"] == "detector/sampled_scores.pkl"
    assert cfg["sparse"]["landmark_prior_weight"] == 0.05
    assert cfg["dense"]["locability_prior_weight"] == 0.05
    assert manifest["score_stats"]["fused_sampled_mean"] == 0.5


def test_boost_fusion_never_decreases_existing_stdloc_scores(tmp_path):
    source_map = tmp_path / "source_map"
    output_map = tmp_path / "soft_map"
    calibration_path = tmp_path / "calib.pt"
    base_cfg = tmp_path / "stdloc.yaml"
    _write_source_map(source_map)
    _write_base_cfg(base_cfg)
    torch.save({"landmark_matchability": torch.tensor([0.1, 0.9], dtype=torch.float32)}, calibration_path)

    soft_prior.build_soft_prior_map(
        source_map=source_map,
        output_map=output_map,
        calibration_path=calibration_path,
        base_cfg_path=base_cfg,
        output_cfg_path=output_map / "stdloc_soft_prior.yaml",
        rho=1.0,
        fusion_mode="boost",
        update_point_cloud_locability=False,
    )

    fused = pickle.load((output_map / "detector/sampled_scores.pkl").open("rb"))
    assert torch.allclose(fused["sampled_scores"], torch.tensor([0.2, 1.0]))
    assert torch.allclose(fused["score_avg"], torch.tensor([0.1, 0.2, 0.3, 1.0]))


def test_load_reliability_from_selfmap_summary_uses_dense_median(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"dense": {"median_te_cm": 8.0, "recall_5cm_5d": 0.4}}', encoding="utf-8")

    reliability = soft_prior.load_selfmap_reliability(summary, stage="dense", center_cm=10.0, temperature_cm=1.0)

    assert reliability["median_te_cm"] == 8.0
    assert reliability["rho"] > 0.85


def test_reliability_can_be_tempered_by_selfmap_recall(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text('{"dense": {"median_te_cm": 8.0, "recall_5cm_5d": 0.4}}', encoding="utf-8")

    median_only = soft_prior.load_selfmap_reliability(summary, stage="dense", center_cm=10.0, temperature_cm=1.0)
    recall_tempered = soft_prior.load_selfmap_reliability(
        summary,
        stage="dense",
        center_cm=10.0,
        temperature_cm=1.0,
        r5_center=0.5,
        r5_temperature=0.1,
    )

    assert recall_tempered["rho"] < median_only["rho"] * 0.5
