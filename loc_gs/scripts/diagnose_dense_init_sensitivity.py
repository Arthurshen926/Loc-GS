#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from loc_gs.diagnostics.match_visualization import (
    draw_match_canvas,
    offset_camera_center,
    pose_error_cm_deg,
    project_points,
    summarize_match_quality,
    write_json,
)
from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle
from loc_gs.scripts.visualize_stdloc_hard_matches import (
    _build_context,
    _capture_dense,
    _capture_sparse,
    _case_rows,
    _inlier_mask,
    _tensor_to_image,
    _write_matches_csv,
)


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _variant_poses(sparse_pose: np.ndarray, gt_w2c: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    return [
        ("sparse_init", "STDLoc sparse-estimated pose", sparse_pose),
        ("gt_oracle_init", "GT pose oracle diagnostic", gt_w2c),
        (
            "gt_shift_050cm",
            "GT camera center shifted by 0.5m in world x",
            offset_camera_center(gt_w2c, np.array([0.5, 0.0, 0.0], dtype=np.float32)),
        ),
        (
            "gt_shift_100cm",
            "GT camera center shifted by 1.0m in world x",
            offset_camera_center(gt_w2c, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ),
    ]


def _summarize_dense(
    *,
    dense: dict[str, Any],
    gt_w2c: np.ndarray,
    good_px: float,
    width: int,
    height: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    dense_gt_xy, dense_gt_valid = project_points(
        dense["p3d"],
        gt_w2c,
        dense["K"],
        width=width,
        height=height,
    )
    dense_errors = (
        np.linalg.norm(dense_gt_xy - dense["query_xy"], axis=1)
        if dense["query_xy"].shape[0]
        else np.empty(0)
    )
    dense_errors[~dense_gt_valid] = np.inf
    dense_inliers = _inlier_mask(len(dense_errors), dense["inliers"])
    dense_good = np.isfinite(dense_errors) & (dense_errors <= good_px)
    return (
        summarize_match_quality(
            reprojection_errors_px=dense_errors,
            solver_inlier_mask=dense_inliers,
            good_px_threshold=good_px,
        ),
        dense_errors,
        dense_inliers,
        dense_good,
    )


def _make_contact_sheet(images: list[tuple[str, Path]], output_path: Path) -> Path:
    loaded: list[tuple[str, Image.Image]] = [
        (label, Image.open(path).convert("RGB")) for label, path in images if path.exists()
    ]
    if not loaded:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), "white").save(output_path)
        return output_path
    width = max(image.width for _label, image in loaded)
    height = sum(image.height + 24 for _label, image in loaded)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for label, image in loaded:
        draw.text((8, y + 5), label, fill=(0, 0, 0))
        y += 24
        canvas.paste(image, (0, y))
        y += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return output_path


def _analyze_case(
    ctx: dict[str, Any],
    row: dict[str, Any],
    output_dir: Path,
    *,
    good_px: float,
    max_draw: int,
) -> dict[str, Any]:
    camera = ctx["camera_by_name"].get(row["image_name"])
    if camera is None:
        raise KeyError(f"camera not found for {ctx['scene']} {row['image_name']}")
    query_image = camera.original_image.to("cuda")
    gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    case_name = f"{ctx['scene']}_{int(row['query_index']):05d}"
    case_dir = output_dir / ctx["scene"] / case_name
    with torch.no_grad():
        sparse = _capture_sparse(ctx, query_image, camera.FoVx, camera.FoVy)
    query_pil = _tensor_to_image(query_image, size=(sparse["width"], sparse["height"]))
    variants = []
    sheet_images: list[tuple[str, Path]] = []
    for variant_name, description, init_pose in _variant_poses(sparse["pose_w2c"], gt_w2c):
        with torch.no_grad():
            dense = _capture_dense(
                ctx,
                sparse["coarse_map"],
                sparse["fine_map"],
                init_pose,
                camera.FoVx,
                camera.FoVy,
            )
        summary, errors, inliers, good = _summarize_dense(
            dense=dense,
            gt_w2c=gt_w2c,
            good_px=good_px,
            width=sparse["width"],
            height=sparse["height"],
        )
        init_te_cm, init_re_deg = pose_error_cm_deg(init_pose, gt_w2c)
        final_te_cm, final_re_deg = pose_error_cm_deg(dense["pose_w2c"], gt_w2c)
        render = (
            _tensor_to_image(dense["render"], size=(sparse["width"], sparse["height"]))
            if dense.get("render") is not None
            else Image.new("RGB", (sparse["width"], sparse["height"]), "black")
        )
        match_path = case_dir / variant_name / "dense_matches.jpg"
        draw_match_canvas(
            query_pil,
            render,
            query_xy=dense["query_xy"],
            reference_xy=dense["rendered_xy"],
            gt_good_mask=good,
            solver_inlier_mask=inliers,
            output_path=match_path,
            max_draw=max_draw,
        )
        _write_matches_csv(
            case_dir / variant_name / "dense_matches.csv",
            dense["query_xy"],
            dense["rendered_xy"],
            errors,
            inliers,
            good,
        )
        variant = {
            "variant": variant_name,
            "description": description,
            "init_te_cm": init_te_cm,
            "init_re_deg": init_re_deg,
            "final_dense_te_cm": final_te_cm,
            "final_dense_re_deg": final_re_deg,
            "dense": summary,
            "paths": {
                "dense_matches": str(match_path),
                "dense_csv": str(case_dir / variant_name / "dense_matches.csv"),
            },
        }
        write_json(case_dir / variant_name / "summary.json", variant)
        variants.append(variant)
        sheet_images.append((variant_name, match_path))
    sparse_te_cm, sparse_re_deg = pose_error_cm_deg(sparse["pose_w2c"], gt_w2c)
    sheet_path = _make_contact_sheet(sheet_images, case_dir / "init_sensitivity_sheet.jpg")
    case_summary = {
        "scene": ctx["scene"],
        "query_index": int(row["query_index"]),
        "image_name": row["image_name"],
        "categories": row.get("categories", ""),
        "diagnostic_only": True,
        "paper_safe_for_tuning": False,
        "captured_sparse_te_cm": sparse_te_cm,
        "captured_sparse_re_deg": sparse_re_deg,
        "variants": variants,
        "paths": {"contact_sheet": str(sheet_path)},
    }
    write_json(case_dir / "init_sensitivity_summary.json", case_summary)
    return case_summary


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene",
                "query_index",
                "image_name",
                "variant",
                "init_te_cm",
                "init_re_deg",
                "final_dense_te_cm",
                "final_dense_re_deg",
                "match_count",
                "gt_good_count",
                "gt_good_ratio",
                "solver_inlier_count",
                "solver_inlier_gt_good_count",
                "solver_inlier_gt_bad_count",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            for variant in summary["variants"]:
                dense = variant["dense"]
                writer.writerow(
                    {
                        "scene": summary["scene"],
                        "query_index": summary["query_index"],
                        "image_name": summary["image_name"],
                        "variant": variant["variant"],
                        "init_te_cm": variant["init_te_cm"],
                        "init_re_deg": variant["init_re_deg"],
                        "final_dense_te_cm": variant["final_dense_te_cm"],
                        "final_dense_re_deg": variant["final_dense_re_deg"],
                        "match_count": dense["match_count"],
                        "gt_good_count": dense["gt_good_count"],
                        "gt_good_ratio": dense["gt_good_ratio"],
                        "solver_inlier_count": dense["solver_inlier_count"],
                        "solver_inlier_gt_good_count": dense["solver_inlier_gt_good_count"],
                        "solver_inlier_gt_bad_count": dense["solver_inlier_gt_bad_count"],
                    }
                )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose dense matching sensitivity to initial pose.")
    parser.add_argument("--report_dir", default="output/reports/cambridge_test_v6_guarded512_20260525")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--output_dir", default="output/diagnostics/dense_init_sensitivity/cambridge_test_v6_guarded512_20260527")
    parser.add_argument("--category", default="hard_failure")
    parser.add_argument("--max_cases", type=int, default=3)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--good_px", type=float, default=5.0)
    parser.add_argument("--max_draw", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    cases = _case_rows(report_dir / "hard_cases.csv", category=str(args.category), max_cases=int(args.max_cases))
    if args.scene:
        allowed = set(args.scene)
        cases = [row for row in cases if row.get("scene") in allowed]
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    candidate_root = Path(args.candidate_root)
    for row in cases:
        scene = str(row["scene"])
        if scene not in contexts:
            contexts[scene] = _build_context(candidate_root / scene)
        summaries.append(
            _analyze_case(
                contexts[scene],
                row,
                output_dir,
                good_px=float(args.good_px),
                max_draw=int(args.max_draw),
            )
        )
    aggregate = {
        "schema": "loc_gs_dense_init_sensitivity_v1",
        "diagnostic_only": True,
        "paper_safe_for_tuning": False,
        "uses_gt_pose_for_oracle_visual_diagnostic": True,
        "report_dir": str(report_dir),
        "candidate_root": str(candidate_root),
        "category": str(args.category),
        "case_count": len(summaries),
        "summaries": summaries,
    }
    write_json(output_dir / "init_sensitivity_summary.json", aggregate)
    _write_summary_csv(output_dir / "init_sensitivity_summary.csv", summaries)
    write_artifact_audit_bundle(
        output_dir,
        manifest={
            "method": "loc_gs_dense_init_sensitivity",
            "diagnostic_only": True,
            "paper_safe_for_tuning": False,
            "split_name": "test",
            "report_dir": str(report_dir),
            "candidate_root": str(candidate_root),
            "output_dir": str(output_dir),
            "case_count": len(summaries),
        },
        command=_command(),
        metrics_summary={"case_count": len(summaries), "diagnostic_only": True},
        split_audit={
            "audit_status": "failed",
            "reason": "official test cases and GT oracle poses are used for diagnosis only; not valid for tuning or model selection",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
