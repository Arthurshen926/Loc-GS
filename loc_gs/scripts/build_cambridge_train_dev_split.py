#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from loc_gs.export.manifest import write_export_eval_audit_bundle
from loc_gs.scripts.export_ulfloc_sparse_feedback import load_dataset_image_list


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        return "unknown"


def _read_rows(path: Path) -> list[str]:
    return [
        raw.strip()
        for raw in Path(path).read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.strip().startswith("#")
    ]


def _row_image(row: str) -> str:
    return row.split()[0]


def split_train_rows(
    rows: Sequence[str],
    *,
    dev_fraction: float,
    min_dev: int,
    max_dev: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not rows:
        raise ValueError("dataset_train contains no image rows")
    if dev_fraction <= 0.0 or dev_fraction >= 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    count = max(int(min_dev), int(round(len(rows) * float(dev_fraction))))
    if max_dev > 0:
        count = min(count, int(max_dev))
    count = min(max(1, count), len(rows) - 1)
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    dev_indices = set(indices[:count])
    selfmap = [row for idx, row in enumerate(rows) if idx not in dev_indices]
    dev = [row for idx, row in enumerate(rows) if idx in dev_indices]
    return selfmap, dev


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a disjoint Cambridge train/selfmap and train-dev split.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dev_fraction", default=0.2, type=float)
    parser.add_argument("--min_dev", default=20, type=int)
    parser.add_argument("--max_dev", default=0, type=int)
    parser.add_argument("--seed", default=13, type=int)
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_argparser().parse_args()

    source_path = Path(args.source_path)
    train_path = source_path / "dataset_train.txt"
    test_path = source_path / "dataset_test.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"dataset_train.txt not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"dataset_test.txt not found: {test_path}")

    train_rows = _read_rows(train_path)
    selfmap_rows, dev_rows = split_train_rows(
        train_rows,
        dev_fraction=float(args.dev_fraction),
        min_dev=int(args.min_dev),
        max_dev=int(args.max_dev),
        seed=int(args.seed),
    )
    selfmap_ids = {_row_image(row) for row in selfmap_rows}
    dev_ids = {_row_image(row) for row in dev_rows}
    official_test_ids = set(load_dataset_image_list(test_path))
    checks = {
        "selfmap_train_dev_disjointness": {
            "status": "failed" if (selfmap_ids & dev_ids) else "passed",
            "overlap": sorted(selfmap_ids & dev_ids),
        },
        "train_dev_official_test_disjointness": {
            "status": "failed" if (dev_ids & official_test_ids) else "passed",
            "overlap": sorted(dev_ids & official_test_ids),
        },
        "selfmap_official_test_disjointness": {
            "status": "failed" if (selfmap_ids & official_test_ids) else "passed",
            "overlap": sorted(selfmap_ids & official_test_ids),
        },
    }
    audit_status = "failed" if any(check["status"] == "failed" for check in checks.values()) else "passed"
    split_audit = {
        "audit_status": audit_status,
        "scene": str(args.scene),
        "source": "dataset_train_heldout",
        "split_name": "train-dev",
        "official_test_used": False,
        "paper_safe_for_tuning": audit_status == "passed",
        "checks": checks,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selfmap_path = output_dir / "selfmap_train.txt"
    dev_path = output_dir / "train_dev.txt"
    selfmap_path.write_text("\n".join(selfmap_rows) + "\n", encoding="utf-8")
    dev_path.write_text("\n".join(dev_rows) + "\n", encoding="utf-8")

    metrics = {
        "train_count": len(train_rows),
        "selfmap_count": len(selfmap_rows),
        "train_dev_count": len(dev_rows),
        "official_test_count": len(official_test_ids),
    }
    manifest = {
        "method": "cambridge_train_dev_split",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "scene": str(args.scene),
        "source_path": str(source_path),
        "dataset_train": str(train_path),
        "dataset_test": str(test_path),
        "selfmap_train_list": str(selfmap_path),
        "train_dev_list": str(dev_path),
        "hyperparameters": {
            "dev_fraction": float(args.dev_fraction),
            "min_dev": int(args.min_dev),
            "max_dev": int(args.max_dev),
            "seed": int(args.seed),
        },
    }
    command = [sys.executable, "-m", "loc_gs.scripts.build_cambridge_train_dev_split", *sys.argv[1:]]
    write_export_eval_audit_bundle(
        output_dir,
        manifest=manifest,
        command=command,
        metrics_summary=metrics,
        split_audit=split_audit,
    )
    if audit_status != "passed":
        raise RuntimeError(f"train-dev split audit failed: {json.dumps(checks, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
