from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from loc_gs.stdloc_native.soft_prior import _reset_path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _mirror_source_without_detector(
    *,
    source_map: Path,
    output_map: Path,
    rebuilt_detector_dir: Path,
) -> int:
    symlinked = 0
    rebuilt_resolved = rebuilt_detector_dir.resolve()
    for src in sorted(source_map.rglob("*")):
        src_resolved = src.resolve()
        if _is_relative_to(src_resolved, rebuilt_resolved):
            continue
        rel = src.relative_to(source_map)
        if rel.parts and rel.parts[0] == "detector":
            continue
        dst = output_map / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src_resolved)
        symlinked += 1
    return symlinked


def _copy_detector_payload(rebuilt_detector_dir: Path, output_detector_dir: Path) -> int:
    files = 0
    output_detector_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(rebuilt_detector_dir.rglob("*")):
        rel = src.relative_to(rebuilt_detector_dir)
        dst = output_detector_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src.resolve() if src.is_symlink() else src, dst)
        files += 1
    return files


def _copy_source_detector_support_files(source_detector_dir: Path, output_detector_dir: Path) -> int:
    if not source_detector_dir.exists():
        return 0
    excluded = {"sampled_idx.pkl", "sampled_scores.pkl"}
    files = 0
    output_detector_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_detector_dir.rglob("*")):
        rel = src.relative_to(source_detector_dir)
        if src.is_dir():
            (output_detector_dir / rel).mkdir(parents=True, exist_ok=True)
            continue
        if rel.as_posix() in excluded:
            continue
        dst = output_detector_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src.resolve() if src.is_symlink() else src, dst)
        files += 1
    return files


def materialize_detector_support_files(
    source_detector_dir: str | Path,
    target_detector_dir: str | Path,
) -> dict[str, Any]:
    """Copy native detector support files while preserving target sampling payload."""

    source_detector_dir = Path(source_detector_dir)
    target_detector_dir = Path(target_detector_dir)
    if not source_detector_dir.exists():
        raise FileNotFoundError(f"source detector dir not found: {source_detector_dir}")
    if not target_detector_dir.exists():
        raise FileNotFoundError(f"target detector dir not found: {target_detector_dir}")
    files = _copy_source_detector_support_files(source_detector_dir, target_detector_dir)
    manifest = {
        "source_detector_dir": str(source_detector_dir),
        "target_detector_dir": str(target_detector_dir),
        "source_detector_support_files_materialized": int(files),
        "sampling_payload_preserved": True,
    }
    (target_detector_dir / "detector_support_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_clean_detector_source_map(
    *,
    source_map: str | Path,
    rebuilt_detector_dir: str | Path,
    output_map: str | Path,
    scene: str = "",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Build a STDLoc source map with a materialized rebuilt detector payload.

    This is intentionally descriptor-neutral: all native map files are mirrored
    unchanged, while ``detector/`` is replaced by a clean detector directory.
    """

    source_map = Path(source_map)
    rebuilt_detector_dir = Path(rebuilt_detector_dir)
    output_map = Path(output_map)
    if not source_map.exists():
        raise FileNotFoundError(f"source map not found: {source_map}")
    if not rebuilt_detector_dir.exists():
        raise FileNotFoundError(f"rebuilt detector dir not found: {rebuilt_detector_dir}")
    sampled_idx_path = rebuilt_detector_dir / "sampled_idx.pkl"
    sampled_scores_path = rebuilt_detector_dir / "sampled_scores.pkl"
    if not sampled_idx_path.exists():
        raise FileNotFoundError(f"rebuilt detector has no sampled_idx.pkl: {sampled_idx_path}")
    if not sampled_scores_path.exists():
        raise FileNotFoundError(f"rebuilt detector has no sampled_scores.pkl: {sampled_scores_path}")
    if output_map.exists() or output_map.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output map already exists: {output_map}")
        _reset_path(output_map)
    output_map.mkdir(parents=True, exist_ok=True)

    symlinked_files = _mirror_source_without_detector(
        source_map=source_map,
        output_map=output_map,
        rebuilt_detector_dir=rebuilt_detector_dir,
    )
    source_detector_support_files = _copy_source_detector_support_files(
        source_map / "detector",
        output_map / "detector",
    )
    materialized_detector_files = _copy_detector_payload(
        rebuilt_detector_dir,
        output_map / "detector",
    )
    sampled_idx = _load_pickle(sampled_idx_path)
    rebuilt_sampled_count = int(len(sampled_idx))

    manifest = {
        "scene": str(scene),
        "source_map": str(source_map),
        "rebuilt_detector_dir": str(rebuilt_detector_dir),
        "output_map": str(output_map),
        "source_detector_excluded": True,
        "rebuilt_sampled_count": rebuilt_sampled_count,
        "source_files_symlinked": int(symlinked_files),
        "source_detector_support_files_materialized": int(source_detector_support_files),
        "detector_files_materialized": int(materialized_detector_files),
        "descriptor_mode": "native",
        "single_path_deployment": True,
        "branch_selection": False,
    }
    (output_map / "clean_source_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
