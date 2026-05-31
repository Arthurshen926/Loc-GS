from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from loc_gs.diagnostics.match_visualization import (
    pose_error_cm_deg,
    project_points,
    summarize_match_quality,
    write_json,
)
from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle
from loc_gs.scripts.diagnose_patch_guided_sparse import _build_context, _capture_sparse_with_overrides, _command


def summarize_sparse_confidence(sparse: Mapping[str, Any]) -> dict[str, Any]:
    query_xy = np.asarray(sparse.get("query_xy", []), dtype=np.float32).reshape(-1, 2)
    inliers = np.asarray(sparse.get("inliers", []), dtype=np.int64).reshape(-1)
    match_count = int(query_xy.shape[0])
    inlier_count = int(inliers.shape[0])
    return {
        "match_count": match_count,
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / max(1, match_count)),
    }


def summarize_sparse_match_quality(
    sparse: Mapping[str, Any],
    *,
    gt_pose_w2c: np.ndarray,
    good_px_threshold: float = 5.0,
) -> dict[str, Any]:
    query_xy = np.asarray(sparse.get("query_xy", []), dtype=np.float32).reshape(-1, 2)
    points = np.asarray(sparse.get("p3d", []), dtype=np.float32).reshape(-1, 3)
    intrinsic = np.asarray(sparse.get("K", np.eye(3, dtype=np.float32)), dtype=np.float32).reshape(3, 3)
    width = int(sparse.get("width", 0) or 0)
    height = int(sparse.get("height", 0) or 0)
    if width <= 0 and query_xy.size:
        width = int(max(1.0, float(np.nanmax(query_xy[:, 0]) + 1.0)))
    if height <= 0 and query_xy.size:
        height = int(max(1.0, float(np.nanmax(query_xy[:, 1]) + 1.0)))
    count = min(int(query_xy.shape[0]), int(points.shape[0]))
    query_xy = query_xy[:count]
    points = points[:count]
    if count == 0:
        return summarize_match_quality(
            reprojection_errors_px=np.empty((0,), dtype=np.float32),
            solver_inlier_mask=np.empty((0,), dtype=bool),
            good_px_threshold=float(good_px_threshold),
        )
    projected, valid = project_points(
        points,
        np.asarray(gt_pose_w2c, dtype=np.float32).reshape(4, 4),
        intrinsic,
        width=max(1, width),
        height=max(1, height),
    )
    errors = np.linalg.norm(projected - query_xy, axis=1)
    errors[~valid] = np.inf
    solver_mask = np.zeros((count,), dtype=bool)
    inliers = np.asarray(sparse.get("inliers", []), dtype=np.int64).reshape(-1)
    inliers = inliers[(inliers >= 0) & (inliers < count)]
    solver_mask[inliers] = True
    return summarize_match_quality(
        reprojection_errors_px=errors,
        solver_inlier_mask=solver_mask,
        good_px_threshold=float(good_px_threshold),
    )


def classify_sparse_failure(
    row: Mapping[str, Any],
    *,
    success_te_cm: float = 50.0,
    min_true_matches: int = 4,
    min_solver_true_inliers: int = 4,
    min_solver_inliers: int = 12,
) -> str:
    """Classify sparse failures using only offline diagnostic GT match quality.

    The labels are for train/self-map failure decomposition. They must not be
    used as test-time branch labels or selector supervision.
    """

    te_cm = float(row.get("sparse_te_cm", np.inf))
    if np.isfinite(te_cm) and te_cm <= float(success_te_cm):
        return "sparse_ok"
    gt_good = int(row.get("gt_good_count", 0) or 0)
    solver_good = int(row.get("solver_inlier_gt_good_count", 0) or 0)
    solver_bad = int(row.get("solver_inlier_gt_bad_count", 0) or 0)
    inlier_count = int(row.get("inlier_count", row.get("solver_inlier_count", 0)) or 0)
    if gt_good < int(min_true_matches):
        return "availability_or_descriptor_no_true_matches"
    if inlier_count < int(min_solver_inliers):
        return "low_solver_support"
    if solver_good < int(min_solver_true_inliers) and solver_bad >= int(min_solver_true_inliers):
        return "false_consensus"
    if solver_good >= int(min_solver_true_inliers) and solver_bad > solver_good:
        return "mixed_true_false_consensus"
    if solver_good >= int(min_solver_true_inliers):
        return "geometry_or_pose_conditioning"
    return "ranking_or_pnp_sampling"


def mine_scene(
    *,
    candidate_root: Path,
    scene: str,
    split: str,
    output_dir: Path,
    max_queries: int,
    low_inlier_threshold: int,
    solver: str | None,
    max_iterations: int | None,
    min_iterations: int | None,
    good_px_threshold: float,
) -> dict[str, Any]:
    ctx = _build_context(candidate_root / scene, split_override=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cameras = list(ctx["cameras"])
    if int(max_queries) > 0:
        cameras = cameras[: int(max_queries)]
    for query_index, camera in enumerate(cameras):
        query_image = camera.original_image.to("cuda")
        gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
        with torch.no_grad():
            sparse = _capture_sparse_with_overrides(
                ctx,
                query_image,
                camera.FoVx,
                camera.FoVy,
                solver=solver,
                max_iterations=max_iterations,
                min_iterations=min_iterations,
            )
        confidence = summarize_sparse_confidence(sparse)
        match_quality = summarize_sparse_match_quality(
            sparse,
            gt_pose_w2c=gt_w2c,
            good_px_threshold=float(good_px_threshold),
        )
        te_cm, re_deg = pose_error_cm_deg(sparse["pose_w2c"], gt_w2c)
        failure_row = {
            **confidence,
            **match_quality,
            "sparse_te_cm": float(te_cm),
            "sparse_re_deg": float(re_deg),
        }
        rows.append(
            {
                "scene": scene,
                "split": split,
                "query_index": int(query_index),
                "image_name": str(camera.image_name),
                **confidence,
                **match_quality,
                "sparse_te_cm": float(te_cm),
                "sparse_re_deg": float(re_deg),
                "low_inlier": bool(confidence["inlier_count"] < int(low_inlier_threshold)),
                "failure_type": classify_sparse_failure(failure_row),
            }
        )
    csv_path = output_dir / f"{scene}_{split}_sparse_confidence.csv"
    fieldnames = [
        "scene",
        "split",
        "query_index",
        "image_name",
        "match_count",
        "inlier_count",
        "inlier_ratio",
        "gt_good_count",
        "gt_bad_count",
        "gt_good_ratio",
        "solver_inlier_gt_good_count",
        "solver_inlier_gt_bad_count",
        "median_reprojection_error_px",
        "p90_reprojection_error_px",
        "sparse_te_cm",
        "sparse_re_deg",
        "low_inlier",
        "failure_type",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    low_rows = [row for row in rows if bool(row["low_inlier"])]
    summary = {
        "schema": "loc_gs_sparse_base_confidence_mining_v1",
        "diagnostic_only": True,
        "candidate_root": str(candidate_root),
        "scene": scene,
        "split": split,
        "query_count": int(len(rows)),
        "low_inlier_threshold": int(low_inlier_threshold),
        "low_inlier_count": int(len(low_rows)),
        "failure_type_counts": {
            key: int(sum(1 for row in rows if str(row.get("failure_type")) == key))
            for key in sorted({str(row.get("failure_type")) for row in rows})
        },
        "csv_path": str(csv_path),
        "lowest_inlier_queries": sorted(rows, key=lambda row: (int(row["inlier_count"]), float(row["sparse_te_cm"])))[:20],
    }
    write_json(output_dir / f"{scene}_{split}_sparse_confidence_summary.json", summary)
    write_artifact_audit_bundle(
        output_dir,
        manifest={
            "method": "sparse_base_confidence_mining",
            "diagnostic_only": True,
            "candidate_root": str(candidate_root),
            "scene": scene,
            "split_name": split,
            "output_dir": str(output_dir),
            "max_queries": int(max_queries),
        "low_inlier_threshold": int(low_inlier_threshold),
        "good_px_threshold": float(good_px_threshold),
        "solver": str(solver) if solver else "native_config",
        },
        command=_command(),
        metrics_summary=summary,
        split_audit={
            "audit_status": "failed" if split == "test" else "unknown",
            "reason": "Sparse confidence mining is valid for train diagnostics; test mining must not be used for tuning.",
        },
    )
    print(json.dumps({"summary": str(output_dir / f"{scene}_{split}_sparse_confidence_summary.json"), "csv": str(csv_path)}, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine sparse-stage base confidence for Cambridge/STDLoc scenes.")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--scene", default="KingsCollege")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output_dir", default="output/diagnostics/sparse_only_pgsh/base_confidence_mining")
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--low_inlier_threshold", type=int, default=80)
    parser.add_argument("--solver", default="native", choices=["native", "opencv", "poselib"])
    parser.add_argument("--max_iterations", type=int, default=0)
    parser.add_argument("--min_iterations", type=int, default=0)
    parser.add_argument("--good_px_threshold", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    mine_scene(
        candidate_root=Path(args.candidate_root),
        scene=str(args.scene),
        split=str(args.split),
        output_dir=Path(args.output_dir),
        max_queries=int(args.max_queries),
        low_inlier_threshold=int(args.low_inlier_threshold),
        solver=None if str(args.solver) == "native" else str(args.solver),
        max_iterations=int(args.max_iterations) if int(args.max_iterations) > 0 else None,
        min_iterations=int(args.min_iterations) if int(args.min_iterations) > 0 else None,
        good_px_threshold=float(args.good_px_threshold),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
