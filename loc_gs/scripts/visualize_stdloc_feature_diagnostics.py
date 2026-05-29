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
import torch.nn.functional as F
from PIL import Image, ImageDraw

from loc_gs.diagnostics.match_visualization import (
    cosine_similarity_map,
    feature_norm_map,
    feature_pair_pca_rgb,
    offset_camera_center,
    pose_error_cm_deg,
    write_json,
)
from loc_gs.reporting.artifact_audit import write_artifact_audit_bundle
from loc_gs.scripts.visualize_stdloc_hard_matches import (
    _build_context,
    _case_rows,
    _tensor_to_image,
)


DIAGNOSTIC_ONLY = True


def _command() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def _to_numpy_chw(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().astype(np.float32)


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    if hi - lo < 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(array, nan=lo) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def _heatmap_rgb(values: np.ndarray) -> Image.Image:
    gray = _normalize_to_uint8(values)
    x = gray.astype(np.float32) / 255.0
    red = np.clip(2.0 * x, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * x - 1.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - x), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8)).convert("RGB")


def _resize_np_rgb(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8)).convert("RGB").resize(size)


def _summary_stats(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "mean": float(finite.mean()),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
    }


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


def _draw_sheet(panels: list[tuple[str, Image.Image]], output_path: Path, *, columns: int = 4) -> Path:
    if not panels:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), "white").save(output_path)
        return output_path
    cell_w = max(image.width for _label, image in panels)
    cell_h = max(image.height for _label, image in panels) + 24
    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        row = idx // columns
        col = idx % columns
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 6, y + 5), label, fill=(0, 0, 0))
        canvas.paste(image, (x, y + 24))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return output_path


def _feature_similarity_maps(query_coarse: torch.Tensor, render_coarse: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    query = F.normalize(query_coarse.float(), dim=0).reshape(query_coarse.shape[0], -1)
    render = F.normalize(render_coarse.float(), dim=0).reshape(render_coarse.shape[0], -1)
    sim = torch.matmul(query.T, render)
    q2r = sim.max(dim=1).values.reshape(query_coarse.shape[-2:]).detach().cpu().numpy().astype(np.float32)
    r2q = sim.max(dim=0).values.reshape(render_coarse.shape[-2:]).detach().cpu().numpy().astype(np.float32)
    return q2r, r2q


def _coarse_mnn_count(ctx: dict[str, Any], query_coarse: torch.Tensor, render_coarse: torch.Tensor) -> int:
    stdloc = ctx["stdloc"]
    loc = ctx["localizer"]
    c, h, w = query_coarse.shape
    corr = torch.matmul(
        F.normalize(query_coarse, dim=0).permute(1, 2, 0).reshape(1, -1, c),
        F.normalize(render_coarse, dim=0).reshape(1, c, -1),
    )
    corr = stdloc.dual_softmax(corr, temp=loc.config["dense"]["coarse_dual_softmax_temp"])
    _b, i, _j = stdloc.mnn_match(corr, thr=loc.config["dense"]["coarse_threshold"])
    return int(i.reshape(-1).numel())


def _render_feature_diagnostics(
    ctx: dict[str, Any],
    query_image: torch.Tensor,
    init_pose: np.ndarray,
    *,
    fovx: float,
    fovy: float,
) -> tuple[dict[str, Any], list[tuple[str, Image.Image]]]:
    stdloc = ctx["stdloc"]
    loc = ctx["localizer"]
    fine_query, coarse_query = loc.get_feature_map(query_image)
    h_f, w_f = fine_query.shape[-2:]
    h_c, w_c = coarse_query.shape[-2:]
    render_pkg = stdloc.render_from_pose_gsplat(
        loc.gaussians,
        torch.tensor(init_pose, device="cuda"),
        fovx,
        fovy,
        w_f,
        h_f,
        render_mode="RGB+ED",
        norm_feat_bf_render=loc.config["dense"]["norm_before_render"],
        rasterize_mode="antialiased",
    )
    fine_render = render_pkg["feature_map"]
    render_rgb = render_pkg.get("render")
    query_rgb = _tensor_to_image(query_image, size=(w_f, h_f))
    if fine_render is None or (fine_render == 0).all():
        blank = Image.new("RGB", (w_f, h_f), "black")
        return (
            {
                "feature_available": False,
                "query_coarse_shape": list(coarse_query.shape),
                "render_coarse_shape": None,
            },
            [("query RGB", query_rgb), ("render RGB", blank)],
        )
    coarse_render = F.interpolate(fine_render[None], size=(h_c, w_c), mode="bilinear", align_corners=False)[0]
    q_np = _to_numpy_chw(coarse_query)
    r_np = _to_numpy_chw(coarse_render)
    q_pca, r_pca = feature_pair_pca_rgb(q_np, r_np)
    same_cosine = cosine_similarity_map(q_np, r_np)
    q2r_max, r2q_max = _feature_similarity_maps(coarse_query, coarse_render)
    query_norm = feature_norm_map(q_np)
    render_norm = feature_norm_map(r_np)
    render_norm_ratio = float((render_norm > 1e-4).mean())
    coarse_mnn_count = _coarse_mnn_count(ctx, coarse_query, coarse_render)
    panel_size = (w_f, h_f)
    panels = [
        ("query RGB", query_rgb),
        (
            "render RGB",
            _tensor_to_image(render_rgb, size=panel_size) if render_rgb is not None else Image.new("RGB", panel_size, "black"),
        ),
        ("query feature PCA", _resize_np_rgb(q_pca, panel_size)),
        ("render feature PCA", _resize_np_rgb(r_pca, panel_size)),
        ("same-pixel cosine", _heatmap_rgb(same_cosine).resize(panel_size)),
        ("query->render max cosine", _heatmap_rgb(q2r_max).resize(panel_size)),
        ("render feature norm", _heatmap_rgb(render_norm).resize(panel_size)),
        ("query feature norm", _heatmap_rgb(query_norm).resize(panel_size)),
    ]
    summary = {
        "feature_available": True,
        "query_coarse_shape": list(coarse_query.shape),
        "render_coarse_shape": list(coarse_render.shape),
        "render_feature_nonzero_ratio": render_norm_ratio,
        "coarse_mnn_count": coarse_mnn_count,
        "same_pixel_cosine": _summary_stats(same_cosine),
        "query_to_render_max_cosine": _summary_stats(q2r_max),
        "render_to_query_max_cosine": _summary_stats(r2q_max),
        "query_feature_norm": _summary_stats(query_norm),
        "render_feature_norm": _summary_stats(render_norm),
    }
    return summary, panels


def _capture_sparse_pose(ctx: dict[str, Any], query_image: torch.Tensor, fovx: float, fovy: float) -> np.ndarray:
    from loc_gs.scripts.visualize_stdloc_hard_matches import _capture_sparse

    sparse = _capture_sparse(ctx, query_image, fovx, fovy)
    return sparse["pose_w2c"]


def _analyze_case(ctx: dict[str, Any], row: dict[str, Any], output_dir: Path, *, max_variants: int | None = None) -> dict[str, Any]:
    camera = ctx["camera_by_name"].get(row["image_name"])
    if camera is None:
        raise KeyError(f"camera not found for {ctx['scene']} {row['image_name']}")
    query_image = camera.original_image.to("cuda")
    gt_w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    with torch.no_grad():
        sparse_pose = _capture_sparse_pose(ctx, query_image, camera.FoVx, camera.FoVy)
    case_name = f"{ctx['scene']}_{int(row['query_index']):05d}"
    case_dir = output_dir / ctx["scene"] / case_name
    summaries = []
    case_panels: list[tuple[str, Image.Image]] = []
    variants = _variant_poses(sparse_pose, gt_w2c)
    if max_variants is not None:
        variants = variants[: int(max_variants)]
    for variant_name, description, init_pose in variants:
        with torch.no_grad():
            feature_summary, panels = _render_feature_diagnostics(
                ctx,
                query_image,
                init_pose,
                fovx=camera.FoVx,
                fovy=camera.FoVy,
            )
        init_te_cm, init_re_deg = pose_error_cm_deg(init_pose, gt_w2c)
        variant_dir = case_dir / variant_name
        sheet_path = _draw_sheet(panels, variant_dir / "feature_diagnostics.jpg", columns=4)
        write_json(
            variant_dir / "feature_summary.json",
            {
                "variant": variant_name,
                "description": description,
                "init_te_cm": init_te_cm,
                "init_re_deg": init_re_deg,
                "diagnostic_only": DIAGNOSTIC_ONLY,
                "feature": feature_summary,
                "paths": {"feature_diagnostics": str(sheet_path)},
            },
        )
        summaries.append(
            {
                "variant": variant_name,
                "description": description,
                "init_te_cm": init_te_cm,
                "init_re_deg": init_re_deg,
                "feature": feature_summary,
                "paths": {"feature_diagnostics": str(sheet_path)},
            }
        )
        case_panels.extend([(f"{variant_name}: {label}", image) for label, image in panels[:4]])
    case_sheet = _draw_sheet(case_panels, case_dir / "feature_case_sheet.jpg", columns=4)
    summary = {
        "scene": ctx["scene"],
        "query_index": int(row["query_index"]),
        "image_name": row["image_name"],
        "categories": row.get("categories", ""),
        "diagnostic_only": DIAGNOSTIC_ONLY,
        "paper_safe_for_tuning": False,
        "variants": summaries,
        "paths": {"feature_case_sheet": str(case_sheet)},
    }
    write_json(case_dir / "feature_case_summary.json", summary)
    return summary


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
                "feature_available",
                "render_feature_nonzero_ratio",
                "coarse_mnn_count",
                "same_pixel_cosine_mean",
                "same_pixel_cosine_p50",
                "query_to_render_max_cosine_mean",
                "query_to_render_max_cosine_p50",
                "render_feature_norm_mean",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            for variant in summary["variants"]:
                feature = variant["feature"]
                writer.writerow(
                    {
                        "scene": summary["scene"],
                        "query_index": summary["query_index"],
                        "image_name": summary["image_name"],
                        "variant": variant["variant"],
                        "init_te_cm": variant["init_te_cm"],
                        "init_re_deg": variant["init_re_deg"],
                        "feature_available": feature.get("feature_available"),
                        "render_feature_nonzero_ratio": feature.get("render_feature_nonzero_ratio"),
                        "coarse_mnn_count": feature.get("coarse_mnn_count"),
                        "same_pixel_cosine_mean": feature.get("same_pixel_cosine", {}).get("mean"),
                        "same_pixel_cosine_p50": feature.get("same_pixel_cosine", {}).get("p50"),
                        "query_to_render_max_cosine_mean": feature.get("query_to_render_max_cosine", {}).get("mean"),
                        "query_to_render_max_cosine_p50": feature.get("query_to_render_max_cosine", {}).get("p50"),
                        "render_feature_norm_mean": feature.get("render_feature_norm", {}).get("mean"),
                    }
                )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize STDLoc dense feature maps for Cambridge hard cases.")
    parser.add_argument("--report_dir", default="output/reports/cambridge_test_v6_guarded512_20260525")
    parser.add_argument("--candidate_root", default="output/stdloc_native/cambridge_test_v6_guarded512_20260525/selected")
    parser.add_argument("--output_dir", default="output/diagnostics/stdloc_feature_diagnostics/cambridge_test_v6_guarded512_20260527")
    parser.add_argument("--category", default="hard_failure")
    parser.add_argument("--max_cases", type=int, default=6)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--query_index", action="append", type=int, default=[])
    parser.add_argument("--max_variants", type=int, default=0, help="0 means all variants")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    cases = _case_rows(report_dir / "hard_cases.csv", category=str(args.category), max_cases=int(args.max_cases))
    if args.scene:
        allowed_scenes = set(args.scene)
        cases = [row for row in cases if row.get("scene") in allowed_scenes]
    if args.query_index:
        allowed_indices = {int(item) for item in args.query_index}
        cases = [row for row in cases if int(row.get("query_index") or -1) in allowed_indices]
    contexts: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for row in cases:
        scene = str(row["scene"])
        if scene not in contexts:
            contexts[scene] = _build_context(Path(args.candidate_root) / scene)
        summaries.append(
            _analyze_case(
                contexts[scene],
                row,
                output_dir,
                max_variants=None if int(args.max_variants) <= 0 else int(args.max_variants),
            )
        )
    write_json(
        output_dir / "feature_diagnostics_summary.json",
        {
            "schema": "loc_gs_stdloc_feature_diagnostics_v1",
            "diagnostic_only": DIAGNOSTIC_ONLY,
            "paper_safe_for_tuning": False,
            "report_dir": str(report_dir),
            "case_count": len(summaries),
            "summaries": summaries,
        },
    )
    _write_summary_csv(output_dir / "feature_diagnostics_summary.csv", summaries)
    write_artifact_audit_bundle(
        output_dir,
        manifest={
            "method": "loc_gs_stdloc_feature_diagnostics",
            "diagnostic_only": DIAGNOSTIC_ONLY,
            "paper_safe_for_tuning": False,
            "split_name": "test",
            "report_dir": str(report_dir),
            "candidate_root": str(args.candidate_root),
            "output_dir": str(output_dir),
            "case_count": len(summaries),
        },
        command=_command(),
        metrics_summary={"case_count": len(summaries), "diagnostic_only": DIAGNOSTIC_ONLY},
        split_audit={
            "audit_status": "failed",
            "reason": "official test cases and GT oracle poses are used for diagnosis only; not valid for tuning or model selection",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
