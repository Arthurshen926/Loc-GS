from pathlib import Path
import struct

import numpy as np

from loc_gs.data.colmap_pose_repair import (
    audit_qvec_norms,
    cambridge_pose_qnorm_issue_map,
    camera_center_from_colmap,
    normalize_qvec,
    read_colmap_image_qvecs,
)


def test_normalizing_colmap_qvec_restores_stmarys_frame174_center() -> None:
    qvec = np.asarray([0.003359, 0.017963, -1.018482, 0.234648], dtype=np.float64)
    expected_center = np.asarray([13.332542, 0.090429, 57.011429], dtype=np.float64)
    raw_tvec = np.asarray([13.245, 25.29, 51.118], dtype=np.float64)

    raw_center = camera_center_from_colmap(qvec, raw_tvec, normalize=False)
    fixed_center = camera_center_from_colmap(qvec, raw_tvec, normalize=True)

    assert np.linalg.norm(raw_center - expected_center) * 100.0 > 800.0
    assert np.allclose(fixed_center, expected_center, atol=5e-3)


def test_audit_qvec_norms_flags_only_outliers() -> None:
    issues = audit_qvec_norms(
        [
            ("seq13/frame00173.png", np.asarray([1.0, 0.0, 0.0, 0.0])),
            ("seq13/frame00174.png", np.asarray([0.0, 0.0, 1.045, 0.0])),
        ],
        tolerance=0.01,
    )

    assert [issue.image_name for issue in issues] == ["seq13/frame00174.png"]
    assert issues[0].qnorm == 1.045


def test_normalize_qvec_rejects_zero_norm() -> None:
    try:
        normalize_qvec(np.zeros(4, dtype=np.float64))
    except ValueError as exc:
        assert "zero-norm" in str(exc)
    else:
        raise AssertionError("zero-norm quaternion was accepted")


def _write_images_bin(path: Path, entries: list[tuple[int, np.ndarray, np.ndarray, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(entries)))
        for image_id, qvec, tvec, name in entries:
            handle.write(struct.pack("<i", int(image_id)))
            handle.write(struct.pack("<4d", *[float(v) for v in qvec]))
            handle.write(struct.pack("<3d", *[float(v) for v in tvec]))
            handle.write(struct.pack("<i", 1))
            handle.write(name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", 0))


def test_read_colmap_image_qvecs_reads_images_bin_without_stdloc_imports(tmp_path: Path) -> None:
    images_bin = tmp_path / "sparse" / "0" / "images.bin"
    _write_images_bin(
        images_bin,
        [
            (1, np.asarray([1.0, 0.0, 0.0, 0.0]), np.zeros(3), "seq/a.png"),
            (2, np.asarray([0.0, 0.0, 1.045, 0.0]), np.zeros(3), "seq/b.png"),
        ],
    )

    records = read_colmap_image_qvecs(images_bin)
    issues = audit_qvec_norms(records, tolerance=0.01)

    assert [issue.image_name for issue in issues] == ["seq/b.png"]


def test_cambridge_pose_qnorm_issue_map_prefers_evaluator_images_bin(tmp_path: Path) -> None:
    scene_root = tmp_path / "StMarysChurch"
    scene_root.mkdir()
    (scene_root / "dataset_test.txt").write_text(
        "seq13/frame00174.png 0 0 0 0.0 0.0 1.045 0.0\n",
        encoding="utf-8",
    )
    _write_images_bin(
        scene_root / "sparse" / "0" / "images.bin",
        [(1, np.asarray([0.0, 0.0, 1.0, 0.0]), np.zeros(3), "seq13/frame00174.png")],
    )

    assert cambridge_pose_qnorm_issue_map(scene_root) == {}
