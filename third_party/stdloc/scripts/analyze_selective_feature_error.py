import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from utils.camera_utils import loadCam
from utils.selective_reconstruction import summarize_locability_error


def _parse_top_ratios(value):
    return [float(item) for item in value.split(",") if item.strip()]


def _parse_model_spec(spec):
    if "=" in spec:
        label, model_path = spec.split("=", 1)
        return label, model_path
    model_path = spec
    return Path(model_path).name, model_path


def _make_dataset_args(args, model_path):
    return SimpleNamespace(
        sh_degree=args.sh_degree,
        source_path=os.path.abspath(args.source_path),
        feature_type=args.feature_type,
        gaussian_type=args.gaussian_type,
        model_path=model_path,
        images=args.images,
        resolution=args.resolution,
        white_background=args.white_background,
        longest_edge=args.longest_edge,
        data_device=args.data_device,
        eval=False,
        speedup=args.speedup,
        norm_before_render=args.norm_before_render,
        render_items=["RGB", "Depth", "Feature Map"],
    )


def _make_gaussians(args):
    if args.gaussian_type == "3dgs":
        return GaussianModel(args.sh_degree)
    if args.gaussian_type == "2dgs":
        return GaussianModel_2dgs(args.sh_degree)
    raise ValueError(f"Unsupported gaussian type: {args.gaussian_type}")


def _select_camera_infos(camera_infos, max_cameras, stride):
    selected = []
    stride = max(1, int(stride or 1))
    for idx, camera_info in enumerate(camera_infos):
        if idx % stride != 0:
            continue
        selected.append((idx, camera_info))
        if max_cameras and len(selected) >= max_cameras:
            break
    return selected


@torch.no_grad()
def analyze_model(args, label, model_path, feature_extractor, top_ratios):
    dataset = _make_dataset_args(args, model_path)
    gaussians = _make_gaussians(args)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
        dataloader_num_workers=0,
        pin_memory=False,
    )
    camera_infos = (
        scene.scene_info.train_cameras
        if args.split == "train"
        else scene.scene_info.test_cameras
    )
    camera_infos = _select_camera_infos(camera_infos, args.max_cameras, args.stride)

    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        device="cuda",
    )

    abs_errors = []
    cosine_errors = []
    locabilities = []
    for camera_idx, camera_info in camera_infos:
        camera = loadCam(dataset, camera_idx, camera_info, scene.resolution_scales[0])
        render_pkg = render_gsplat(
            camera,
            gaussians,
            background,
            rgb_only=False,
            norm_feat_bf_render=args.norm_before_render,
            longest_edge=args.longest_edge,
            rasterize_mode="antialiased",
        )
        feature_map = render_pkg["feature_map"]
        locability_map = render_pkg.get("locability_map")
        if feature_map is None or locability_map is None:
            continue

        image = camera.original_image.to("cuda")
        gt_feature = feature_extractor(image[None])["feature_map"][0]
        gt_feature = F.interpolate(
            gt_feature[None],
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0]
        gt_feature = F.normalize(gt_feature, p=2, dim=0)

        abs_error = (feature_map - gt_feature).abs().mean(dim=0)
        cosine_error = 1.0 - (feature_map * gt_feature).sum(dim=0).clamp(-1.0, 1.0)
        locability = locability_map.squeeze(0)
        valid = torch.isfinite(abs_error) & torch.isfinite(locability)
        valid = valid & (feature_map.abs().sum(dim=0) > 0)
        if not valid.any():
            continue

        abs_errors.append(abs_error[valid].float().cpu())
        cosine_errors.append(cosine_error[valid].float().cpu())
        locabilities.append(locability[valid].float().cpu())

    if not abs_errors:
        return {
            "label": label,
            "model_path": model_path,
            "num_cameras": len(camera_infos),
            "num_valid_pixels": 0,
            "rows": [],
        }

    abs_errors = torch.cat(abs_errors)
    cosine_errors = torch.cat(cosine_errors)
    locabilities = torch.cat(locabilities)
    abs_rows = summarize_locability_error(abs_errors, locabilities, top_ratios)
    cosine_rows = summarize_locability_error(cosine_errors, locabilities, top_ratios)

    rows = []
    for abs_row, cosine_row in zip(abs_rows, cosine_rows):
        row = dict(abs_row)
        row.update(
            {
                "cosine_all_error_mean": cosine_row["all_error_mean"],
                "cosine_selected_error_mean": cosine_row["selected_error_mean"],
                "cosine_background_error_mean": cosine_row["background_error_mean"],
                "cosine_selected_to_background_error_ratio": cosine_row[
                    "selected_to_background_error_ratio"
                ],
            }
        )
        rows.append(row)

    return {
        "label": label,
        "model_path": model_path,
        "num_cameras": len(camera_infos),
        "num_valid_pixels": int(abs_errors.numel()),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max_cameras", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--top_ratios", default="0.05,0.1,0.2")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--longest_edge", type=int, default=640)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument(
        "--white_background",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument(
        "--norm_before_render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    top_ratios = _parse_top_ratios(args.top_ratios)
    feature_extractor = FeatureExtractor(args.feature_type).cuda().eval()
    models = [_parse_model_spec(item) for item in args.model]

    payload = {
        "source_path": os.path.abspath(args.source_path),
        "iteration": args.iteration,
        "split": args.split,
        "max_cameras": args.max_cameras,
        "stride": args.stride,
        "top_ratios": top_ratios,
        "models": [],
    }
    for label, model_path in models:
        result = analyze_model(args, label, model_path, feature_extractor, top_ratios)
        payload["models"].append(result)

    print(
        f"{'model':18s} {'rho':>6s} {'abs_all':>9s} {'abs_sel':>9s} "
        f"{'abs_bg':>9s} {'abs_ratio':>9s} {'cos_sel':>9s} {'score_mass':>10s}"
    )
    for model in payload["models"]:
        for row in model["rows"]:
            print(
                f"{model['label'][:18]:18s} {row['top_ratio']:6.2f} "
                f"{row['all_error_mean']:9.6f} "
                f"{row['selected_error_mean']:9.6f} "
                f"{row['background_error_mean']:9.6f} "
                f"{row['selected_to_background_error_ratio']:9.4f} "
                f"{row['cosine_selected_error_mean']:9.6f} "
                f"{row['score_mass']:10.4f}"
            )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
