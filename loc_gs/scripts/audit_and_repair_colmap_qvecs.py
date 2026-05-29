#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from loc_gs.data.colmap_pose_repair import audit_qvec_norms, normalize_qvec


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_colmap_io() -> tuple[Any, Any]:
    stdloc_root = _repo_root() / "third_party" / "stdloc"
    datasets_root = stdloc_root / "datasets"
    sys.path.insert(0, str(stdloc_root))
    sys.path.insert(0, str(datasets_root))
    from scene.colmap_loader import read_extrinsics_binary  # type: ignore
    from colmap_from_nvm import write_images_binary  # type: ignore

    return read_extrinsics_binary, write_images_binary


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True).strip()
    except Exception:
        return "unknown"


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _prepare_repaired_scene_root(source_scene_root: Path, output_scene_root: Path, *, overwrite: bool) -> None:
    if output_scene_root.exists() or output_scene_root.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output scene root already exists: {output_scene_root}")
        if output_scene_root.is_symlink() or output_scene_root.is_file():
            output_scene_root.unlink()
        else:
            shutil.rmtree(output_scene_root)
    output_scene_root.mkdir(parents=True, exist_ok=True)

    for child in sorted(source_scene_root.iterdir()):
        if child.name == "sparse":
            continue
        _symlink_or_copy(child, output_scene_root / child.name)

    src_sparse = source_scene_root / "sparse" / "0"
    dst_sparse = output_scene_root / "sparse" / "0"
    dst_sparse.mkdir(parents=True, exist_ok=True)
    for child in sorted(src_sparse.iterdir()):
        if child.name == "images.bin":
            continue
        _symlink_or_copy(child, dst_sparse / child.name)


def _repair_images_bin(source_scene_root: Path, output_scene_root: Path, *, tolerance: float) -> dict[str, Any]:
    read_extrinsics_binary, write_images_binary = _import_colmap_io()
    images_path = source_scene_root / "sparse" / "0" / "images.bin"
    images = read_extrinsics_binary(str(images_path))
    issues = audit_qvec_norms(((image.name, image.qvec) for image in images.values()), tolerance=tolerance)
    issue_names = {issue.image_name for issue in issues}

    repaired = {}
    for image_id, image in images.items():
        if image.name in issue_names:
            qvec = normalize_qvec(np.asarray(image.qvec, dtype=np.float64))
            repaired[image_id] = image._replace(qvec=qvec)
        else:
            repaired[image_id] = image

    write_images_binary(repaired, str(output_scene_root / "sparse" / "0" / "images.bin"))
    return {
        "images_bin": str(images_path),
        "image_count": len(images),
        "qnorm_tolerance": float(tolerance),
        "normalized_count": len(issues),
        "issues": [
            {
                "image_name": issue.image_name,
                "qnorm": round(float(issue.qnorm), 9),
                "delta": round(float(issue.delta), 9),
            }
            for issue in issues
        ],
    }


def audit_and_optionally_repair(
    *,
    scene_root: Path,
    output_scene_root: Path | None,
    tolerance: float = 0.01,
    repair: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    scene_root = scene_root.resolve()
    if not (scene_root / "sparse" / "0" / "images.bin").exists():
        raise FileNotFoundError(f"COLMAP images.bin not found under {scene_root / 'sparse' / '0'}")

    read_extrinsics_binary, _ = _import_colmap_io()
    images = read_extrinsics_binary(str(scene_root / "sparse" / "0" / "images.bin"))
    issues = audit_qvec_norms(((image.name, image.qvec) for image in images.values()), tolerance=tolerance)
    payload: dict[str, Any] = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_scene_root": str(scene_root),
        "output_scene_root": str(output_scene_root.resolve()) if output_scene_root is not None else None,
        "qnorm_tolerance": float(tolerance),
        "image_count": len(images),
        "issue_count": len(issues),
        "issues": [
            {
                "image_name": issue.image_name,
                "qnorm": round(float(issue.qnorm), 9),
                "delta": round(float(issue.delta), 9),
            }
            for issue in issues
        ],
        "repair_written": False,
    }
    if repair:
        if output_scene_root is None:
            raise ValueError("--repair requires --output-scene-root")
        output_scene_root = output_scene_root.resolve()
        _prepare_repaired_scene_root(scene_root, output_scene_root, overwrite=overwrite)
        repair_payload = _repair_images_bin(scene_root, output_scene_root, tolerance=tolerance)
        payload.update(repair_payload)
        payload["output_scene_root"] = str(output_scene_root)
        payload["repair_written"] = True
        (output_scene_root / "pose_qnorm_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (output_scene_root / "manifest.json").write_text(
            json.dumps(
                {
                    "git_commit": payload["git_commit"],
                    "command": " ".join(sys.argv),
                    "source_scene_root": str(scene_root),
                    "output_scene_root": str(output_scene_root),
                    "repair_type": "normalize_colmap_images_bin_qvec_only",
                    "qnorm_tolerance": float(tolerance),
                    "normalized_count": int(payload["normalized_count"]),
                    "timestamp_utc": payload["timestamp_utc"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Cambridge/STDLoc COLMAP images.bin quaternion norms and optionally create a repaired scene root."
    )
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--output-scene-root", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--repair", action="store_true", help="Write a non-destructive repaired scene root.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    payload = audit_and_optionally_repair(
        scene_root=args.scene_root,
        output_scene_root=args.output_scene_root,
        tolerance=float(args.tolerance),
        repair=bool(args.repair),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
