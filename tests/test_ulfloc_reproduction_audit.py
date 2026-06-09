from __future__ import annotations

import json
from pathlib import Path

from loc_gs.reporting.ulfloc_reproduction_audit import (
    build_ulfloc_reproduction_audit,
    classify_ulfloc_reproduction_gaps,
    extract_pose_metrics,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_extract_pose_metrics_accepts_summary_and_metrics_summary_shapes() -> None:
    summary_metrics = extract_pose_metrics(
        {
            "sparse": {"median_te": 13.0, "median_ae": 0.2, "recall_10cm_5d": 0.4},
            "dense": {"median_te": 8.0, "median_ae": 0.1, "recall_10cm_5d": 0.6},
        }
    )
    metrics_summary = extract_pose_metrics(
        {
            "sparse": {"median_te_cm": 12.0, "median_re_deg": 0.3, "R10cm5deg": 0.5},
            "dense": {"median_te_cm": 7.0, "median_re_deg": 0.15, "R10cm5deg": 0.7},
        }
    )

    assert summary_metrics["dense"]["median_te_cm"] == 8.0
    assert summary_metrics["dense"]["median_re_deg"] == 0.1
    assert summary_metrics["dense"]["R10cm5deg"] == 0.6
    assert metrics_summary["sparse"]["median_te_cm"] == 12.0
    assert metrics_summary["dense"]["R10cm5deg"] == 0.7


def test_classify_ulfloc_reproduction_gaps_marks_train_dev_and_local_modifications() -> None:
    report = {
        "checkout": {"missing_files": [], "has_local_modifications": True},
        "environment": {"package_versions": {"gsplat": "unknown"}},
        "results": {
            "scene_count": 1,
            "official_test_scene_count": 0,
            "train_dev_scene_count": 1,
            "best_dense_avg_te_cm": None,
        },
        "config": {"uses_processed_images": False, "mask_config_present": False},
    }

    ids = {gap["id"] for gap in classify_ulfloc_reproduction_gaps(report)}

    assert "ulf_checkout_has_local_modifications" in ids
    assert "missing_five_scene_official_test_reproduction" in ids
    assert "only_train_dev_or_diagnostic_outputs_found" in ids
    assert "unknown_gsplat_version" in ids
    assert "processed_images_or_masks_not_confirmed" in ids


def test_build_ulfloc_reproduction_audit_scans_local_results(tmp_path: Path) -> None:
    ulf_root = tmp_path / "ULF-Loc"
    for rel in (
        "README.md",
        "requirements.txt",
        "train.py",
        "ulfloc.py",
        "configs/ulfloc_cambridge.yaml",
        "scripts/train_cambridge.sh",
        "scripts/evaluate_cambridge.sh",
    ):
        path = ulf_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy\n", encoding="utf-8")
    (ulf_root / "configs/ulfloc_cambridge.yaml").write_text(
        "images: processed\nuse_masks: true\n", encoding="utf-8"
    )

    result_dir = ulf_root / "outputs" / "ShopFacade" / "train_dev"
    _write_json(result_dir / "manifest.json", {"scene": "ShopFacade", "split": "train_dev"})
    _write_json(
        result_dir / "summary.json",
        {"dense": {"median_te": 3.0, "median_ae": 0.13, "recall_10cm_5d": 0.9}},
    )

    report = build_ulfloc_reproduction_audit(ulf_root=ulf_root, result_root=ulf_root / "outputs")

    assert report["checkout"]["checkout_exists"] is True
    assert report["results"]["scene_count"] == 1
    assert report["results"]["train_dev_scene_count"] == 1
    assert report["results"]["best_dense_avg_te_cm"] == 3.0
    assert report["config"]["uses_processed_images"] is True
