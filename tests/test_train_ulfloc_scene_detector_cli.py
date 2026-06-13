import json
import subprocess
import sys

import torch

from loc_gs.scripts.train_ulfloc_scene_detector import (
    _cache_file_for_query,
    _load_or_extract_cached_feature_map,
)


class _FakeFeatureExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def detectAndComputeDense(self, image):
        self.calls += 1
        feature_map = torch.full((1, 256, 3, 4), float(self.calls), dtype=torch.float32)
        return feature_map, torch.zeros((1, 1, 24, 32), dtype=torch.float32)


def test_detector_feature_cache_reuses_query_feature_map(tmp_path):
    extractor = _FakeFeatureExtractor()
    image = torch.zeros(3, 24, 32, dtype=torch.float32)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    cache_dir = tmp_path / "cache"

    first = _load_or_extract_cached_feature_map(
        query_id="seq/frame00001.png",
        image_path=image_path,
        image=image,
        feature_extractor=extractor,
        feature_type="sp",
        cache_dir=cache_dir,
        cache_mode="read_write",
    )
    second = _load_or_extract_cached_feature_map(
        query_id="seq/frame00001.png",
        image_path=image_path,
        image=image,
        feature_extractor=extractor,
        feature_type="sp",
        cache_dir=cache_dir,
        cache_mode="read_write",
    )

    assert extractor.calls == 1
    assert torch.allclose(first, second)
    assert tuple(first.shape) == (256, 3, 4)
    assert _cache_file_for_query(cache_dir, "seq/frame00001.png").exists()


def test_detector_feature_cache_rebuild_mode_ignores_existing_cache(tmp_path):
    extractor = _FakeFeatureExtractor()
    image = torch.zeros(3, 24, 32, dtype=torch.float32)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    cache_dir = tmp_path / "cache"

    _load_or_extract_cached_feature_map(
        query_id="q.png",
        image_path=image_path,
        image=image,
        feature_extractor=extractor,
        feature_type="sp",
        cache_dir=cache_dir,
        cache_mode="read_write",
    )
    rebuilt = _load_or_extract_cached_feature_map(
        query_id="q.png",
        image_path=image_path,
        image=image,
        feature_extractor=extractor,
        feature_type="sp",
        cache_dir=cache_dir,
        cache_mode="rebuild",
    )

    assert extractor.calls == 2
    assert float(rebuilt.mean().item()) == 2.0


def test_detector_feature_cache_can_store_float16_and_read_float32(tmp_path):
    extractor = _FakeFeatureExtractor()
    image = torch.zeros(3, 24, 32, dtype=torch.float32)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    cache_dir = tmp_path / "cache"

    first = _load_or_extract_cached_feature_map(
        query_id="q.png",
        image_path=image_path,
        image=image,
        feature_extractor=extractor,
        feature_type="sp",
        cache_dir=cache_dir,
        cache_mode="read_write",
        cache_dtype="float16",
    )
    payload = torch.load(_cache_file_for_query(cache_dir, "q.png"), map_location="cpu")
    second = _load_or_extract_cached_feature_map(
        query_id="q.png",
        image_path=image_path,
        image=image,
        feature_extractor=extractor,
        feature_type="sp",
        cache_dir=cache_dir,
        cache_mode="readonly",
        cache_dtype="float16",
    )

    assert payload["feature_map"].dtype == torch.float16
    assert payload["metadata"]["cache_dtype"] == "float16"
    assert first.dtype == torch.float32
    assert second.dtype == torch.float32
    assert torch.allclose(first, second)


def test_train_ulfloc_scene_detector_cli_rejects_test_targets(tmp_path):
    targets = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "split_name": "test",
            "targets": {},
            "metadata": {},
            "split_audit": {"split_name": "test", "test_split_used": True},
        },
        targets,
    )
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_scene_detector",
            "--detector_targets",
            str(targets),
            "--source_path",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--epochs",
            "0",
            "--seed",
            "123",
            "--device",
            "cpu",
            "--ulf_root",
            "/root/ULF-Loc",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test split" in result.stderr


def test_train_ulfloc_scene_detector_cli_zero_epoch_writes_audit_bundle(tmp_path):
    targets = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "schema_version": "ulfloc_detector_targets_from_solver_feedback_v1",
            "split_name": "train_self_map",
            "targets": {},
            "metadata": {
                "height": 20,
                "width": 30,
                "detector_target_storage": "points",
                "positive_detector_point_count": 0,
            },
            "split_audit": {"audit_status": "passed", "split_name": "train_self_map", "test_split_used": False},
        },
        targets,
    )
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_scene_detector",
            "--detector_targets",
            str(targets),
            "--source_path",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--epochs",
            "0",
            "--seed",
            "123",
            "--solver_validity_power",
            "1.0",
            "--solver_feedback_residual_alpha",
            "0.05",
            "--device",
            "cpu",
            "--ulf_root",
            "/root/ULF-Loc",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "scene_detector.pth").exists()
    assert (out / "manifest.json").exists()
    assert (out / "metrics_summary.json").exists()
    assert (out / "split_audit.json").exists()
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["trained_image_count"] == 0
    assert metrics["epochs"] == 0
    assert metrics["seed"] == 123
    assert metrics["solver_validity_power"] == 1.0
    assert metrics["solver_feedback_residual_alpha"] == 0.05
    assert metrics["teacher_preserving_residual"] is False
    assert metrics["scene_detector_mode"] == "stdloc_fullres"
    assert metrics["target_mode"] == "stdloc_fullres"
    assert metrics["loss_terms"] == [
        "stdloc_bce_loss",
        "solver_positive_residual_loss",
        "solver_negative_suppression_loss",
    ]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 123
    assert manifest["solver_validity_power"] == 1.0
    assert manifest["solver_feedback_residual_alpha"] == 0.05
    checkpoint = torch.load(out / "scene_detector.pth", map_location="cpu")
    assert checkpoint["metadata"]["scene_detector_mode"] == "stdloc_fullres"
    assert checkpoint["metadata"]["target_mode"] == "stdloc_fullres"
    assert checkpoint["metadata"]["solver_feedback_residual_alpha"] == 0.05


def test_train_ulfloc_scene_detector_cli_zero_epoch_accepts_suppression_targets(tmp_path):
    targets = tmp_path / "detector_targets.pt"
    torch.save(
        {
            "schema_version": "ulfloc_solver_feedback_detector_targets_v1",
            "split_name": "train_selfmap",
            "targets": {
                "q1.png": {
                    "gaussian_ids": torch.tensor([1], dtype=torch.long),
                    "keypoint_yx": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
                    "support_weights": torch.tensor([1.0], dtype=torch.float32),
                    "negative_gaussian_ids": torch.tensor([2], dtype=torch.long),
                    "negative_keypoint_yx": torch.tensor([[30.0, 40.0]], dtype=torch.float32),
                    "negative_weights": torch.tensor([2.0], dtype=torch.float32),
                    "positive_count": 1,
                    "negative_count": 1,
                }
            },
            "metadata": {
                "height": 64,
                "width": 80,
                "detector_target_storage": "points",
                "positive_detector_point_count": 1,
                "negative_suppression_point_count": 1,
            },
            "split_audit": {"audit_status": "passed", "split_name": "train_selfmap", "test_split_used": False},
        },
        targets,
    )
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "loc_gs.scripts.train_ulfloc_scene_detector",
            "--detector_targets",
            str(targets),
            "--source_path",
            str(tmp_path),
            "--output_dir",
            str(out),
            "--epochs",
            "0",
            "--seed",
            "123",
            "--device",
            "cpu",
            "--ulf_root",
            "/root/ULF-Loc",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    checkpoint = torch.load(out / "scene_detector.pth", map_location="cpu")
    assert checkpoint["metadata"]["source_detector_targets"] == str(targets)
