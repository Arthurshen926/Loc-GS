from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from loc_gs.localization.pose_metrics import pose_error_summary


def _summary(errors_te: Iterable[float], errors_ae: Iterable[float], inliers: Iterable[int]) -> dict[str, float]:
    return pose_error_summary(errors_te, errors_ae, inliers)


def merge_eval_dirs(input_dirs: list[Path], output_dir: Path) -> dict[str, object]:
    if not input_dirs:
        raise ValueError("at least one input directory is required")
    summaries = []
    details = []
    seen_images: set[str] = set()
    for input_dir in input_dirs:
        summary_path = input_dir / "summary.json"
        results_path = input_dir / "results.json"
        if not summary_path.exists() or not results_path.exists():
            raise FileNotFoundError(f"missing summary/results in {input_dir}")
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        for row in json.loads(results_path.read_text(encoding="utf-8")):
            name = str(row.get("image_name", ""))
            if name in seen_images:
                raise ValueError(f"duplicate image_name in merged results: {name}")
            seen_images.add(name)
            details.append(row)

    details.sort(key=lambda row: str(row.get("image_name", "")))
    sparse_rows = [row for row in details if row.get("sparse_te") is not None and row.get("sparse_ae") is not None]
    dense_rows = [row for row in details if row.get("dense_te") is not None and row.get("dense_ae") is not None]
    base = dict(summaries[0])
    base["query_offset"] = 0
    base["query_stride"] = 1
    base["sparse"] = _summary(
        (row["sparse_te"] for row in sparse_rows),
        (row["sparse_ae"] for row in sparse_rows),
        (int(row.get("sparse_inliers", 0)) for row in sparse_rows),
    )
    base["dense"] = _summary(
        (row["dense_te"] for row in dense_rows),
        (row["dense_ae"] for row in dense_rows),
        (int(row.get("dense_inliers", 0)) for row in dense_rows),
    )
    base["localized"] = len(dense_rows)
    base["queries"] = len(details)
    base["merged_from"] = [str(path) for path in input_dirs]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return base


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge split Cambridge hybrid eval outputs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("input_dirs", nargs="+")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = merge_eval_dirs([Path(p) for p in args.input_dirs], Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
