#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


CSV_COLUMNS = [
    "query_id",
    "native_sparse_te_cm",
    "stdloc_sparse_te_cm",
    "sparse_delta_te_cm",
    "native_sparse_re_deg",
    "stdloc_sparse_re_deg",
    "sparse_delta_re_deg",
    "native_sparse_matches",
    "stdloc_sparse_matches",
    "native_sparse_inliers",
    "stdloc_sparse_inliers",
    "native_sparse_inlier_ratio",
    "stdloc_sparse_inlier_ratio",
    "native_sparse_median_reproj_px",
    "stdloc_sparse_median_reproj_px",
    "native_dense_te_cm",
    "stdloc_dense_te_cm",
    "dense_delta_te_cm",
    "native_dense_re_deg",
    "stdloc_dense_re_deg",
    "dense_delta_re_deg",
    "native_dense_inliers",
    "stdloc_dense_inliers",
    "native_dense_rejections",
    "stdloc_dense_rejections",
]


def _load_result_rows(path_or_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"results.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"expected a list of query results in {path}")
    rows = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        if not any(row.get(key) is not None for key in ("query_id", "image_name", "name", "query", "image")):
            row = dict(row)
            row["query_id"] = f"query_{index:06d}"
        rows.append(row)
    return rows


def _query_id(row: dict[str, Any]) -> str:
    for key in ("query_id", "image_name", "name", "query", "image"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError(f"query result row has no query identifier: {row}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        current: Any = row
        found = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            value = _as_float(current)
            if value is not None:
                return value
    return None


def _stage_value(row: dict[str, Any], stage: str, metric: str) -> float | None:
    aliases = {
        "te": (
            f"{stage}_TE",
            f"{stage}_te",
            f"{stage}_te_cm",
            f"{stage}.TE",
            f"{stage}.te",
            f"{stage}.te_cm",
            f"{stage}.median_te_cm",
        ),
        "re": (
            f"{stage}_AE",
            f"{stage}_ae",
            f"{stage}_re",
            f"{stage}_re_deg",
            f"{stage}.AE",
            f"{stage}.ae",
            f"{stage}.re",
            f"{stage}.re_deg",
        ),
        "inliers": (f"{stage}_inliers", f"{stage}.inliers", f"{stage}_num_inliers"),
        "matches": (
            f"{stage}_matches",
            f"{stage}_match_count",
            f"{stage}.matches",
            f"{stage}.match_count",
            "matchability.sparse_match_count" if stage == "sparse" else "",
            "oracle_match.oracle_match_count" if stage == "sparse" else "",
            "oracle_match.candidate_oracle_pnp_match_count" if stage == "sparse" else "",
        ),
        "median_reproj": (
            f"{stage}_median_reproj_px",
            f"{stage}.median_reproj_px",
            "oracle_match.oracle_reproj_median_px" if stage == "sparse" else "",
            "oracle_match.candidate_oracle_reproj_median_px" if stage == "sparse" else "",
            "matchability.sparse_all_reproj_median_px" if stage == "sparse" else "",
        ),
        "rejections": (f"{stage}_rejections", f"{stage}.rejections"),
    }
    return _first_float(row, *(key for key in aliases[metric] if key))


def _delta(native: float | None, stdloc: float | None) -> float | None:
    if native is None or stdloc is None:
        return None
    return native - stdloc


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0.0:
        return None
    return num / den


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return str(float(value))


def _run_summary(rows: list[dict[str, Any]], stage: str) -> dict[str, float]:
    te = [row[f"{stage}_te_cm"] for row in rows if row.get(f"{stage}_te_cm") is not None]
    re = [row[f"{stage}_re_deg"] for row in rows if row.get(f"{stage}_re_deg") is not None]
    paired = [
        (row[f"{stage}_te_cm"], row[f"{stage}_re_deg"])
        for row in rows
        if row.get(f"{stage}_te_cm") is not None and row.get(f"{stage}_re_deg") is not None
    ]
    out: dict[str, float] = {"localized": float(len(te))}
    if te:
        out["median_te_cm"] = float(median(te))
        out["mean_te_cm"] = float(sum(te) / len(te))
    if re:
        out["median_re_deg"] = float(median(re))
        out["mean_re_deg"] = float(sum(re) / len(re))
    if paired:
        out["recall_5cm_5deg"] = float(sum(1 for t, r in paired if t <= 5.0 and r <= 5.0) / len(rows))
        out["recall_10cm_5deg"] = float(sum(1 for t, r in paired if t <= 10.0 and r <= 5.0) / len(rows))
    return out


def audit_cambridge_parity(
    *,
    native_dir: str | Path,
    stdloc_dir: str | Path,
    output_dir: str | Path,
    scene: str = "",
) -> dict[str, Any]:
    native_rows = {_query_id(row): row for row in _load_result_rows(native_dir)}
    stdloc_rows = {_query_id(row): row for row in _load_result_rows(stdloc_dir)}
    common = sorted(set(native_rows) & set(stdloc_rows))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows: list[dict[str, Any]] = []
    with (out_dir / "parity_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for query in common:
            native = native_rows[query]
            stdloc = stdloc_rows[query]
            native_sparse_matches = _stage_value(native, "sparse", "matches")
            stdloc_sparse_matches = _stage_value(stdloc, "sparse", "matches")
            row = {
                "query_id": query,
                "native_sparse_te_cm": _stage_value(native, "sparse", "te"),
                "stdloc_sparse_te_cm": _stage_value(stdloc, "sparse", "te"),
                "native_sparse_re_deg": _stage_value(native, "sparse", "re"),
                "stdloc_sparse_re_deg": _stage_value(stdloc, "sparse", "re"),
                "native_sparse_matches": native_sparse_matches,
                "stdloc_sparse_matches": stdloc_sparse_matches,
                "native_sparse_inliers": _stage_value(native, "sparse", "inliers"),
                "stdloc_sparse_inliers": _stage_value(stdloc, "sparse", "inliers"),
                "native_sparse_median_reproj_px": _stage_value(native, "sparse", "median_reproj"),
                "stdloc_sparse_median_reproj_px": _stage_value(stdloc, "sparse", "median_reproj"),
                "native_dense_te_cm": _stage_value(native, "dense", "te"),
                "stdloc_dense_te_cm": _stage_value(stdloc, "dense", "te"),
                "native_dense_re_deg": _stage_value(native, "dense", "re"),
                "stdloc_dense_re_deg": _stage_value(stdloc, "dense", "re"),
                "native_dense_inliers": _stage_value(native, "dense", "inliers"),
                "stdloc_dense_inliers": _stage_value(stdloc, "dense", "inliers"),
                "native_dense_rejections": _stage_value(native, "dense", "rejections"),
                "stdloc_dense_rejections": _stage_value(stdloc, "dense", "rejections"),
            }
            row["sparse_delta_te_cm"] = _delta(row["native_sparse_te_cm"], row["stdloc_sparse_te_cm"])
            row["sparse_delta_re_deg"] = _delta(row["native_sparse_re_deg"], row["stdloc_sparse_re_deg"])
            row["dense_delta_te_cm"] = _delta(row["native_dense_te_cm"], row["stdloc_dense_te_cm"])
            row["dense_delta_re_deg"] = _delta(row["native_dense_re_deg"], row["stdloc_dense_re_deg"])
            row["native_sparse_inlier_ratio"] = _ratio(row["native_sparse_inliers"], native_sparse_matches)
            row["stdloc_sparse_inlier_ratio"] = _ratio(row["stdloc_sparse_inliers"], stdloc_sparse_matches)
            normalized_rows.append(row)
            writer.writerow({key: row["query_id"] if key == "query_id" else _fmt(row.get(key)) for key in CSV_COLUMNS})

    summary: dict[str, Any] = {
        "scene": str(scene),
        "native_dir": str(native_dir),
        "stdloc_dir": str(stdloc_dir),
        "common_queries": len(common),
        "native_only_queries": len(set(native_rows) - set(stdloc_rows)),
        "stdloc_only_queries": len(set(stdloc_rows) - set(native_rows)),
        "native": {
            "sparse": _run_summary(
                [
                    {"sparse_te_cm": row["native_sparse_te_cm"], "sparse_re_deg": row["native_sparse_re_deg"]}
                    for row in normalized_rows
                ],
                "sparse",
            ),
            "dense": _run_summary(
                [
                    {"dense_te_cm": row["native_dense_te_cm"], "dense_re_deg": row["native_dense_re_deg"]}
                    for row in normalized_rows
                ],
                "dense",
            ),
        },
        "stdloc": {
            "sparse": _run_summary(
                [
                    {"sparse_te_cm": row["stdloc_sparse_te_cm"], "sparse_re_deg": row["stdloc_sparse_re_deg"]}
                    for row in normalized_rows
                ],
                "sparse",
            ),
            "dense": _run_summary(
                [
                    {"dense_te_cm": row["stdloc_dense_te_cm"], "dense_re_deg": row["stdloc_dense_re_deg"]}
                    for row in normalized_rows
                ],
                "dense",
            ),
        },
    }
    native_dense = summary["native"]["dense"]
    stdloc_dense = summary["stdloc"]["dense"]
    native_sparse = summary["native"]["sparse"]
    stdloc_sparse = summary["stdloc"]["sparse"]
    summary["delta"] = {
        "sparse_median_te_cm": _delta(native_sparse.get("median_te_cm"), stdloc_sparse.get("median_te_cm")),
        "dense_median_te_cm": _delta(native_dense.get("median_te_cm"), stdloc_dense.get("median_te_cm")),
        "sparse_recall_5cm_5deg": _delta(
            native_sparse.get("recall_5cm_5deg"),
            stdloc_sparse.get("recall_5cm_5deg"),
        ),
        "dense_recall_5cm_5deg": _delta(
            native_dense.get("recall_5cm_5deg"),
            stdloc_dense.get("recall_5cm_5deg"),
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write query-level native-vs-STDLoc Cambridge parity diagnostics.")
    parser.add_argument("--native_dir", required=True)
    parser.add_argument("--stdloc_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = audit_cambridge_parity(
        native_dir=args.native_dir,
        stdloc_dir=args.stdloc_dir,
        output_dir=args.output_dir,
        scene=args.scene,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
