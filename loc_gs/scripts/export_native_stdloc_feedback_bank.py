#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from loc_gs.feedback.audit import audit_feedback_bank_v2
from loc_gs.feedback.io import save_feedback_bank, summarize_feedback_bank
from loc_gs.stdloc_native.commands import resolve_scene_images
from loc_gs.stdloc_native.native_feedback import (
    dense_transition_from_errors,
    records_from_native_sparse_capture,
    reprojection_errors_px,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STDLOC_ROOT = REPO_ROOT / "third_party" / "stdloc"


def _prepare_stdloc_imports() -> Any:
    stdloc_path = str(STDLOC_ROOT)
    if stdloc_path not in sys.path:
        sys.path.insert(0, stdloc_path)
    import stdloc as stdloc_module

    return stdloc_module


def _repo_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else REPO_ROOT / raw


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_status() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(REPO_ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return f"git unavailable: {exc}"
    return result.stdout.strip()


def _command_from_argv() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, "-m", "loc_gs.scripts.export_native_stdloc_feedback_bank", *sys.argv[1:]])


def _json_default(value: Any) -> Any:
    import torch

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _load_sampled_idx(model_path: Path, config: dict[str, Any]) -> Any:
    import torch

    rel = config.get("sparse", {}).get("landmark_path", "detector/sampled_idx.pkl")
    path = model_path / str(rel)
    if not path.exists():
        raise FileNotFoundError(f"missing sampled_idx for native feedback export: {path}")
    return torch.as_tensor(_load_pickle(path), dtype=torch.long).reshape(-1).cpu()


def _resolve_cfg(path: str | Path) -> Path:
    raw = _repo_path(path)
    if raw.exists():
        return raw
    vendored = STDLOC_ROOT / Path(path)
    if vendored.exists():
        return vendored
    raise FileNotFoundError(f"STDLoc cfg not found: {path}")


def _resolve_scene_images_for_export(scene_root: str | Path, preferred: str) -> str:
    root = Path(scene_root)
    if root.exists() and preferred and not (root / preferred).exists():
        return "."
    return preferred


def _build_dataset(stdloc_module: Any, args: argparse.Namespace, scene_root: Path, model_path: Path) -> Any:
    parser = argparse.ArgumentParser(add_help=False)
    model = stdloc_module.ModelParams(parser, sentinel=False)
    stdloc_module.PipelineParams(parser)
    parsed = parser.parse_args(
        [
            "-s",
            str(scene_root),
            "-m",
            str(model_path),
            "-r",
            str(int(args.resolution)),
            "-f",
            str(args.feature_type),
            "-g",
            str(args.gaussian_type),
            "--images",
            str(_resolve_scene_images_for_export(scene_root, args.images)),
            "--data_device",
            str(args.data_device),
        ]
    )
    return model.extract(parsed)


def _match_ranks(query_indices: list[int], scores: list[float]) -> list[int]:
    grouped: dict[int, list[tuple[int, float]]] = {}
    for index, (query_idx, score) in enumerate(zip(query_indices, scores)):
        grouped.setdefault(int(query_idx), []).append((index, float(score)))
    ranks = [1 for _ in scores]
    for entries in grouped.values():
        for rank, (index, _score) in enumerate(sorted(entries, key=lambda item: (-item[1], item[0])), start=1):
            ranks[index] = rank
    return ranks


def _captured_sparse_localize(
    stdloc_module: Any,
    stdloc: Any,
    query_feature_map: Any,
    fovx: float,
    fovy: float,
    sampled_idx: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    height, width = query_feature_map.shape[-2:]
    heat_map = stdloc.detector(query_feature_map)
    kp_scores_after_nms = stdloc_module.simple_nms(
        heat_map,
        stdloc.config["sparse"].get("nms", 4),
    ).flatten()
    _, kp_ids = torch.topk(
        kp_scores_after_nms,
        stdloc.config["sparse"].get("detect_num", 2048),
    )
    pos_mask = kp_scores_after_nms > 0
    kp_ids = kp_ids[pos_mask[kp_ids]]
    kp_mask = torch.zeros_like(kp_scores_after_nms, dtype=torch.bool)
    kp_mask[kp_ids] = True

    sampled_features = query_feature_map.reshape(query_feature_map.shape[0], -1)[:, kp_mask]
    landmark_features = F.normalize(stdloc.landmarks.get_loc_feature.squeeze(), dim=-1)
    corr_matrix = torch.matmul(sampled_features.T, landmark_features.T)
    corr_matrix = stdloc_module.apply_landmark_prior(
        corr_matrix,
        stdloc.landmark_prior,
        weight=stdloc.config["sparse"].get("landmark_prior_weight", 0.0),
    )
    if stdloc.config["sparse"]["dual_softmax"] is True:
        corr_matrix = stdloc_module.dual_softmax(
            corr_matrix=corr_matrix,
            temp=stdloc.config["sparse"]["dual_softmax_temp"],
        )
    if stdloc.config["sparse"]["mnn_match"] is True:
        _batch_ids, im_idx, gs_ids = stdloc_module.mnn_match(
            corr_matrix[None],
            thr=stdloc.config["sparse"]["threshold"],
        )
        match_scores = corr_matrix[im_idx, gs_ids]
    else:
        im_idx, gs_ids, match_scores = stdloc_module.topk_match(
            corr_matrix[None],
            stdloc.config["sparse"]["topk"],
            thr=stdloc.config["sparse"]["threshold"],
        )

    all_xy = torch.stack(
        [torch.arange(height * width) % width, torch.arange(height * width) // width],
        dim=1,
    )
    matched_xy = all_xy[kp_mask.cpu()][im_idx.cpu()].numpy().astype(np.float64) + 0.5
    p3d = stdloc.landmarks.get_xyz[gs_ids].detach().cpu().numpy()
    intrinsic = stdloc_module.get_intrinsic(fovx, fovy, width, height)
    pose_w2c, inliers = stdloc_module.solve_pose(
        matched_xy,
        p3d,
        intrinsic,
        stdloc.config["sparse"]["solver"],
        stdloc.config["sparse"]["reprojection_error"],
        stdloc.config["sparse"]["confidence"],
        stdloc.config["sparse"]["max_iterations"],
        stdloc.config["sparse"]["min_iterations"],
        match_scores=match_scores.detach().cpu().numpy(),
    )
    inliers = np.asarray(inliers).reshape(-1).astype(np.int64)
    scores = [float(item) for item in match_scores.detach().cpu().reshape(-1).tolist()]
    query_indices = [int(item) for item in im_idx.detach().cpu().reshape(-1).tolist()]
    selected_kp_ids = kp_ids.detach().cpu()[im_idx.detach().cpu()].reshape(-1)
    detector_scores = kp_scores_after_nms.detach().cpu()[selected_kp_ids].reshape(-1)
    if stdloc.landmark_prior is None:
        visibility_scores = [None for _ in scores]
    else:
        visibility_scores = [
            float(item)
            for item in stdloc.landmark_prior.detach().cpu()[gs_ids.detach().cpu()].reshape(-1).tolist()
        ]
    full_gaussian_ids = sampled_idx[gs_ids.detach().cpu()].reshape(-1)
    errors = reprojection_errors_px(matched_xy, p3d, intrinsic, pose_w2c)
    capture = {
        "keypoint_indices": [int(item) for item in selected_kp_ids.tolist()],
        "keypoint_xy": [(float(x), float(y)) for x, y in matched_xy.tolist()],
        "landmark_ids": [int(item) for item in gs_ids.detach().cpu().reshape(-1).tolist()],
        "gaussian_ids": [int(item) for item in full_gaussian_ids.tolist()],
        "descriptor_scores": scores,
        "detector_scores": [float(item) for item in detector_scores.tolist()],
        "match_ranks": _match_ranks(query_indices, scores),
        "inlier_indices": [int(item) for item in inliers.tolist()],
        "reprojection_errors_px": errors,
        "visibility_scores": visibility_scores,
    }
    return {"pose_w2c": pose_w2c, "inliers": int(inliers.shape[0])}, capture


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export audited feedback_bank_v2 from native STDLoc train sparse matches.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--map_root", required=True)
    parser.add_argument("--map_scene", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cfg", default="configs/stdloc_cambridge_opencv.yaml")
    parser.add_argument("--images", default="processed")
    parser.add_argument("--feature_type", default="sp")
    parser.add_argument("--gaussian_type", default="3dgs")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval_split", choices=["train", "test"], default="train")
    parser.add_argument("--max_test_cameras", type=int, default=80)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--split_name", default="selfmap_train_native_stdloc")
    parser.add_argument("--split_audit_json", required=True)
    parser.add_argument("--te_epsilon_cm", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if str(args.eval_split).strip().lower() == "test":
        raise ValueError("test split is not allowed for native feedback bank export")
    if str(args.split_name).strip().lower() == "test":
        raise ValueError("test split_name is not allowed for native feedback bank export")


def export_feedback_bank(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    stdloc_module = _prepare_stdloc_imports()
    output = _repo_path(args.output_dir)
    scene_root = _repo_path(Path(args.data_root) / args.scene)
    map_scene = args.map_scene or args.scene
    model_path = _repo_path(Path(args.map_root) / map_scene)
    split_audit = json.loads(_repo_path(args.split_audit_json).read_text(encoding="utf-8"))
    if split_audit.get("audit_status") != "passed":
        raise ValueError("split_audit_json must have audit_status=passed")
    cfg_path = _resolve_cfg(args.cfg)
    if args.dry_run:
        return {
            "dry_run": True,
            "scene": args.scene,
            "scene_root": str(scene_root),
            "map_path": str(model_path),
            "cfg": str(cfg_path),
            "images": _resolve_scene_images_for_export(scene_root, args.images),
            "output_dir": str(output),
            "split_name": args.split_name,
        }

    dataset = _build_dataset(stdloc_module, args, scene_root, model_path)
    if dataset.gaussian_type == "3dgs":
        gaussians = stdloc_module.GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = stdloc_module.GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")
    scene = stdloc_module.Scene(
        dataset,
        gaussians,
        load_iteration=-1,
        shuffle=False,
        preload_cameras=False,
        dataloader_num_workers=0,
        pin_memory=False,
    )

    config = yaml.load(cfg_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    sampled_idx = _load_sampled_idx(model_path, config)
    stdloc = stdloc_module.STDLoc(gaussians, config)

    eval_cameras = scene.getTrainCameras()
    cameras = stdloc_module.select_eval_cameras(
        eval_cameras,
        max_cameras=int(args.max_test_cameras) if int(args.max_test_cameras) > 0 else None,
        stride=int(args.test_stride),
    )
    records = []
    query_summaries: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for camera_info in cameras:
        gt_w2c = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
        query_image = camera_info.original_image.to("cuda")
        fine_map, coarse_map = stdloc.get_feature_map(query_image)
        sparse_result, capture = _captured_sparse_localize(
            stdloc_module,
            stdloc,
            fine_map,
            camera_info.FoVx,
            camera_info.FoVy,
            sampled_idx,
        )
        pose_w2c = sparse_result["pose_w2c"]
        dense_results = []
        for _iter in range(int(config["dense"]["iters"])):
            dense_result = stdloc.loc_dense(
                coarse_map,
                fine_map,
                pose_w2c,
                camera_info.FoVx,
                camera_info.FoVy,
            )
            pose_w2c = dense_result["pose_w2c"]
            dense_results.append(dense_result)
        sparse_ae, sparse_te = stdloc_module.cal_pose_error(sparse_result["pose_w2c"], gt_w2c)
        dense_ae, dense_te = stdloc_module.cal_pose_error(dense_results[-1]["pose_w2c"], gt_w2c)
        dense_transition, dense_delta = dense_transition_from_errors(
            sparse_te_cm=float(sparse_te),
            dense_te_cm=float(dense_te),
            te_epsilon_cm=float(args.te_epsilon_cm),
        )
        records.extend(
            records_from_native_sparse_capture(
                scene=args.scene,
                image_name=str(camera_info.image_name),
                sparse_te_cm=float(sparse_te),
                sparse_re_deg=float(sparse_ae),
                dense_te_cm=float(dense_te),
                dense_re_deg=float(dense_ae),
                dense_transition=dense_transition,
                dense_delta_te_cm=dense_delta,
                **capture,
            )
        )
        query_summaries.append(
            {
                "image_name": str(camera_info.image_name),
                "match_count": int(len(capture["gaussian_ids"])),
                "sparse_inliers": int(sparse_result["inliers"]),
                "sparse_te_cm": float(sparse_te),
                "sparse_re_deg": float(sparse_ae),
                "dense_te_cm": float(dense_te),
                "dense_re_deg": float(dense_ae),
                "dense_transition": dense_transition,
                "dense_delta_te_cm": dense_delta,
            }
        )

    manifest = {
        "scene": args.scene,
        "schema_version": "feedback_bank_v2",
        "split_name": args.split_name,
        "query_id_source": "image_id",
        "split_audit": split_audit,
        "source": "export_native_stdloc_feedback_bank",
        "map_path": str(model_path),
        "data_root": str(scene_root),
        "cfg": str(cfg_path),
        "eval_split": args.eval_split,
        "max_test_cameras": int(args.max_test_cameras),
        "test_stride": int(args.test_stride),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": _command_from_argv(),
        "single_path_deployment": True,
        "branch_selection": False,
    }
    bank_path = output / "feedback_bank.jsonl"
    save_feedback_bank(bank_path, records, manifest)
    audit = audit_feedback_bank_v2(bank_path)
    if audit["audit_status"] != "passed":
        raise ValueError(f"exported native feedback bank failed audit: {audit['reasons']}")
    feedback_summary = summarize_feedback_bank(bank_path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output / "feedback_bank_v2_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (output / "feedback_summary.json").write_text(json.dumps(feedback_summary, indent=2, sort_keys=True), encoding="utf-8")
    (output / "query_summaries.json").write_text(json.dumps(query_summaries, indent=2, sort_keys=True), encoding="utf-8")
    (output / "command.txt").write_text(_command_from_argv() + "\n", encoding="utf-8")
    (output / "git_status.txt").write_text(_git_status() + "\n", encoding="utf-8")
    torch.cuda.empty_cache()
    return {
        "dry_run": False,
        "feedback_bank": str(bank_path),
        "feedback_summary": str(output / "feedback_summary.json"),
        "audit": str(output / "feedback_bank_v2_audit.json"),
        "record_count": int(len(records)),
        "query_count": int(len(query_summaries)),
    }


def main(args: argparse.Namespace | None = None) -> int:
    args = build_argparser().parse_args() if args is None else args
    result = export_feedback_bank(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
