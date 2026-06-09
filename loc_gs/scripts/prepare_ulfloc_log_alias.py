#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REQUIRED_FILES = ("keypoints_sampled_idx.pkl", "keypoints_features.pkl")
DEFAULT_OPTIONAL_FILES = ("solver_feedback.pkl",)


def _git_text(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - best-effort provenance
        return f"unavailable: {exc}"


def _command(argv: Sequence[str] | None = None) -> str:
    parts = [sys.executable, "-m", "loc_gs.scripts.prepare_ulfloc_log_alias"]
    parts.extend(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in parts)


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        raise IsADirectoryError(f"refusing to replace directory: {path}")


def _materialize_alias(source: Path, target: Path, *, copy: bool, force: bool) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"{target} already exists; pass --force to replace it")
        _remove_existing(target)
    if copy:
        shutil.copy2(source, target)
        mode = "copy"
    else:
        target.symlink_to(source.resolve())
        mode = "symlink"
    return {"name": target.name, "source": str(source), "target": str(target), "mode": mode}


def prepare_ulfloc_log_alias(
    *,
    source_log_dir: Path,
    target_log_dir: Path,
    required_files: Sequence[str] = DEFAULT_REQUIRED_FILES,
    optional_files: Sequence[str] = DEFAULT_OPTIONAL_FILES,
    copy: bool = False,
    force: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    source_log_dir = source_log_dir.resolve()
    target_log_dir.mkdir(parents=True, exist_ok=True)
    if source_log_dir == target_log_dir.resolve():
        raise ValueError("source_log_dir and target_log_dir must be different")

    linked: list[dict[str, Any]] = []
    missing_optional: list[str] = []
    for name in required_files:
        linked.append(
            _materialize_alias(source_log_dir / name, target_log_dir / name, copy=copy, force=force)
        )
    for name in optional_files:
        source = source_log_dir / name
        if source.exists():
            linked.append(_materialize_alias(source, target_log_dir / name, copy=copy, force=force))
        else:
            missing_optional.append(name)

    manifest = {
        "schema": "loc_gs_ulfloc_log_alias_v1",
        "method": "prepare_ulfloc_log_alias",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_text(["rev-parse", "HEAD"]),
        "command": command or _command(),
        "source_log_dir": str(source_log_dir),
        "target_log_dir": str(target_log_dir),
        "required_files": list(required_files),
        "optional_files": list(optional_files),
        "missing_optional_files": missing_optional,
        "files": linked,
        "copy": bool(copy),
        "force": bool(force),
    }
    (target_log_dir / "alias_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target_log_dir / "alias_command.txt").write_text((command or _command()) + "\n", encoding="utf-8")
    (target_log_dir / "alias_git_status.txt").write_text(_git_text(["status", "--short"]) + "\n", encoding="utf-8")
    return manifest


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Link or copy ULF-Loc sampled keypoint files into a separate eval log directory."
    )
    parser.add_argument("--source_log_dir", required=True, type=Path)
    parser.add_argument("--target_log_dir", required=True, type=Path)
    parser.add_argument("--required_file", action="append", default=None)
    parser.add_argument("--optional_file", action="append", default=None)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    parser.add_argument("--force", action="store_true", help="Replace existing files/symlinks in target_log_dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    manifest = prepare_ulfloc_log_alias(
        source_log_dir=args.source_log_dir,
        target_log_dir=args.target_log_dir,
        required_files=tuple(args.required_file or DEFAULT_REQUIRED_FILES),
        optional_files=tuple(args.optional_file or DEFAULT_OPTIONAL_FILES),
        copy=bool(args.copy),
        force=bool(args.force),
        command=_command(argv),
    )
    print(json.dumps({"target_log_dir": manifest["target_log_dir"], "file_count": len(manifest["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
