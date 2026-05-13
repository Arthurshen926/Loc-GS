#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from loc_gs.data.cambridge_dataset import CambridgeHybridDataset
from loc_gs.localization.stdloc_detector import StdlocKeypointDetector, extract_stdloc_detector_keypoints
from loc_gs.localization.scene_matcher import label_scene_match_pairs
from loc_gs.localization.stdloc_parity import apply_match_prior
from loc_gs.losses.cross_view import projective_view_overlap
from loc_gs.losses.localization_loss import project_world_to_image_yx
from loc_gs.models.hybrid_gaussian import HybridFeatureGaussian, SuperPointOutputHead
from loc_gs.scripts.eval_cambridge_hybrid import (
    build_stdloc_detector_landmark_bank,
    extract_keypoints_from_detector_logits,
    load_cambridge_rgb_no_resize,
    prepare_query_teacher_maps,
    sample_descriptors_bilinear,
    superpoint_gray,
    upsample_feature_map,
)
from loc_gs.scripts.extract_superpoint_features import SuperPointNet
from loc_gs.scripts.train_cambridge_hybrid import (
    pose_delta_trans_rot,
    render_hybrid_superpoint,
    sample_rehearsal_pose_batch,
)


def matchability_from_counts(
    tp_count: torch.Tensor,
    fp_count: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Beta-smoothed estimate of PnP-useful landmark reliability."""
    tp = tp_count.float().clamp_min(0.0)
    fp = fp_count.float().clamp_min(0.0)
    a = max(float(alpha), 0.0)
    return ((tp + a) / (tp + fp + 2.0 * a).clamp_min(1e-6)).clamp(0.0, 1.0)


def calibration_query_canvas_hw(
    resized_rgb: torch.Tensor,
    teacher_rgb: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Return the image canvas used for query keypoints and camera intrinsics.

    SuperPoint can be extracted from an original-resolution teacher image, but
    CambridgeHybridDataset intrinsics and eval-time PnP operate on the resized
    dataset frame.  Calibration labels must therefore use the resized canvas.
    """

    del teacher_rgb
    return int(resized_rgb.shape[-2]), int(resized_rgb.shape[-1])


def _camera_depth(points: torch.Tensor, pose_w2c: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)
    cam = (pose_w2c.to(device=points.device, dtype=points.dtype) @ pts_h.T).T
    return cam[:, 2]


def _sample_scalar_map_bilinear(value_map: torch.Tensor, yx: torch.Tensor) -> torch.Tensor:
    if value_map.dim() == 2:
        value_map = value_map.unsqueeze(0)
    if value_map.dim() != 3 or value_map.shape[0] != 1:
        value_map = value_map.reshape(1, int(value_map.shape[-2]), int(value_map.shape[-1]))
    _, h, w = value_map.shape
    y = yx[:, 0].float()
    x = yx[:, 1].float()
    x_norm = 2.0 * x / max(w - 1, 1) - 1.0
    y_norm = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        value_map.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1)


def write_matchability_calibration(payload: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    for key in (
        "landmark_matchability",
        "landmark_tp_count",
        "landmark_fp_count",
        "landmark_fp_rate",
        "matchability_calibrator",
    ):
        if key in payload:
            torch.save(payload[key], output_path.parent / f"{key}.pt")

    meta = dict(payload.get("metadata", {}))
    matchability = torch.as_tensor(payload["landmark_matchability"]).float()
    tp = torch.as_tensor(payload["landmark_tp_count"]).float()
    fp = torch.as_tensor(payload["landmark_fp_count"]).float()
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                **meta,
                "mean_matchability": float(matchability.mean().item()),
                "observed_landmarks": int(((tp + fp) > 0).sum().item()),
                "total_tp": float(tp.sum().item()),
                "total_fp": float(fp.sum().item()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def accumulate_matchability_counts(
    tp_count: torch.Tensor,
    fp_count: torch.Tensor,
    *,
    query_yx: torch.Tensor,
    landmark_xyz: torch.Tensor,
    q_ids: torch.Tensor,
    lm_ids: torch.Tensor,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    reprojection_threshold_px: float,
    depth_map: torch.Tensor | None = None,
    alpha_map: torch.Tensor | None = None,
    depth_abs_tolerance: float = 0.25,
    depth_rel_tolerance: float = 0.02,
    alpha_threshold: float = 0.05,
) -> None:
    """Accumulate TP/FP labels for top-k query-to-landmark candidates."""
    if q_ids.numel() == 0 or lm_ids.numel() == 0:
        return
    device = landmark_xyz.device
    q = q_ids.to(device=device, dtype=torch.long).reshape(-1)
    lm = lm_ids.to(device=device, dtype=torch.long).reshape(-1)
    pts = landmark_xyz[lm]
    proj_yx, valid_z = project_world_to_image_yx(
        pts.unsqueeze(0),
        pose_w2c.to(device=device, dtype=torch.float32).unsqueeze(0),
        K.to(device=device, dtype=torch.float32),
    )
    err = torch.linalg.norm(proj_yx[0] - query_yx.to(device=device, dtype=torch.float32)[q], dim=-1)
    valid = valid_z[0] & torch.isfinite(err)
    if depth_map is not None or alpha_map is not None:
        proj = proj_yx[0]
        h = int((depth_map if depth_map is not None else alpha_map).shape[-2])
        w = int((depth_map if depth_map is not None else alpha_map).shape[-1])
        in_frame = (proj[:, 0] >= 0.0) & (proj[:, 0] <= h - 1) & (proj[:, 1] >= 0.0) & (proj[:, 1] <= w - 1)
        valid = valid & in_frame
        if depth_map is not None:
            sampled_depth = _sample_scalar_map_bilinear(depth_map.to(device=device), proj)
            z = _camera_depth(pts.float(), pose_w2c.to(device=device, dtype=torch.float32))
            depth_tol = max(float(depth_abs_tolerance), 0.0) + max(float(depth_rel_tolerance), 0.0) * z.abs()
            depth_ok = (sampled_depth > 0.0) & torch.isfinite(sampled_depth) & ((sampled_depth - z).abs() <= depth_tol)
            valid = valid & depth_ok
        if alpha_map is not None:
            sampled_alpha = _sample_scalar_map_bilinear(alpha_map.to(device=device), proj)
            valid = valid & torch.isfinite(sampled_alpha) & (sampled_alpha >= float(alpha_threshold))
    is_tp = valid & (err <= float(reprojection_threshold_px))
    ones = torch.ones_like(err, dtype=tp_count.dtype, device=tp_count.device)
    if is_tp.any():
        tp_count.scatter_add_(0, lm[is_tp].to(tp_count.device), ones[is_tp].to(tp_count.device))
    is_fp = ~is_tp
    if is_fp.any():
        fp_count.scatter_add_(0, lm[is_fp].to(fp_count.device), ones[is_fp].to(fp_count.device))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate landmark matchability from train-view TP/FP matches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--data_root", default="/mnt/pool/sqy/Cambridge_stdloc")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--max_views", type=int, default=128)
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--max_landmarks", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--reprojection_threshold_px", type=float, default=8.0)
    parser.add_argument("--smoothing_alpha", type=float, default=1.0)
    parser.add_argument("--visibility_check", choices=["none", "rendered"], default="rendered")
    parser.add_argument("--visibility_alpha_threshold", type=float, default=0.05)
    parser.add_argument("--visibility_depth_abs_tolerance", type=float, default=0.25)
    parser.add_argument("--visibility_depth_rel_tolerance", type=float, default=0.02)
    parser.add_argument("--query_detector", choices=["superpoint", "stdloc", "feedback"], default="stdloc")
    parser.add_argument("--query_feature_source", choices=["resized", "original"], default="original")
    parser.add_argument("--feedback_detector_full_res", action="store_true")
    parser.add_argument("--query_keypoints", type=int, default=2048)
    parser.add_argument("--keypoint_threshold", type=float, default=0.015)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument(
        "--descriptor_source",
        choices=["hybrid", "ply_loc", "hybrid_ply_blend", "hybrid_ply_gated_residual"],
        default="ply_loc",
    )
    parser.add_argument("--ply_loc_feature_weight", type=float, default=0.9)
    parser.add_argument("--hybrid_residual_alpha_max", type=float, default=0.05)
    parser.add_argument("--locability_prior_weight", type=float, default=0.05)
    parser.add_argument("--scene_match_pair_output_path", default="")
    parser.add_argument("--scene_match_pair_sample_limit", type=int, default=200000)
    parser.add_argument(
        "--scene_match_pair_train_fraction",
        type=float,
        default=1.0,
        help="Fraction of the pair cache reserved for real train-view labels; the rest is reserved for rendered rehearsal views.",
    )
    parser.add_argument("--rendered_rehearsal_views", type=int, default=0)
    parser.add_argument(
        "--rendered_query_source",
        choices=["rendered_rgb_teacher", "feature_field"],
        default="rendered_rgb_teacher",
        help="Use rendered RGB plus the frozen query extractor for query-like calibration, or the rendered field descriptors.",
    )
    parser.add_argument(
        "--rendered_rehearsal_pose_mode",
        choices=["perturb", "pair", "interpolate", "mixed"],
        default="mixed",
    )
    parser.add_argument("--rendered_rehearsal_pair_probability", type=float, default=0.5)
    parser.add_argument("--rendered_rehearsal_interpolation_min", type=float, default=-0.15)
    parser.add_argument("--rendered_rehearsal_interpolation_max", type=float, default=1.15)
    parser.add_argument("--rendered_pose_noise_trans_m", type=float, default=0.35)
    parser.add_argument("--rendered_pose_noise_rot_deg", type=float, default=20.0)
    parser.add_argument("--rendered_pair_jitter_trans_m", type=float, default=0.08)
    parser.add_argument("--rendered_pair_jitter_rot_deg", type=float, default=5.0)
    parser.add_argument("--rendered_view_pair_min_overlap", type=float, default=0.15)
    parser.add_argument("--stdloc_detector_path", default="")
    parser.add_argument("--feedback_detector_path", default="")
    parser.add_argument("--stdloc_detector_dir", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    args = build_argparser().parse_args() if args is None else args
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt["args"]
    scene = args.scene or train_args["scene"]
    scene_root = Path(args.data_root) / scene
    feature_intr = train_args["feature_intrinsics"]
    full_fx = feature_intr["fx"] * 8.0
    full_fy = feature_intr["fy"] * 8.0
    full_cx = feature_intr["cx"] * 8.0
    full_cy = feature_intr["cy"] * 8.0

    model = HybridFeatureGaussian(
        latent_dim=train_args["latent_dim"],
        hash_output_dim=train_args["hash_output_dim"],
        fine_dim=train_args["fine_dim"],
        coarse_dim=train_args["coarse_dim"],
        output_dim=train_args["hybrid_output_dim"],
    )
    model.load_from_ply(train_args["ply_path"])
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    sp_head = SuperPointOutputHead(
        fused_dim=train_args["hybrid_output_dim"],
        descriptor_dim=256,
        detector_dim=65,
        hidden_dim=256,
        num_res_blocks=2,
        use_3x3=True,
    ).to(device)
    sp_head.load_state_dict(ckpt["sp_head_state_dict"])
    sp_head.eval()
    teacher = SuperPointNet().to(device)
    teacher.load_state_dict(torch.load(train_args["superpoint_weights"], map_location=device), strict=False)
    teacher.eval()

    stdloc_detector = None
    feedback_detector = None
    if args.query_detector == "stdloc":
        detector_path = (
            Path(args.stdloc_detector_path)
            if args.stdloc_detector_path
            else Path(train_args["ply_path"]).parents[2] / "detector" / "30000_detector.pth"
        )
        stdloc_detector = StdlocKeypointDetector(in_dim=256).to(device)
        stdloc_detector.load_state_dict(torch.load(detector_path, map_location=device))
        stdloc_detector.eval()
    elif args.query_detector == "feedback":
        feedback_detector = StdlocKeypointDetector(in_dim=256).to(device)
        if args.feedback_detector_path:
            feedback_detector.load_state_dict(torch.load(Path(args.feedback_detector_path), map_location=device))
        elif "feedback_detector_state_dict" in ckpt:
            feedback_detector.load_state_dict(ckpt["feedback_detector_state_dict"], strict=False)
        else:
            raise FileNotFoundError(
                "query_detector=feedback requires --feedback_detector_path or feedback_detector_state_dict in checkpoint"
            )
        feedback_detector.eval()

    detector_dir = (
        Path(args.stdloc_detector_dir)
        if args.stdloc_detector_dir
        else Path(train_args["ply_path"]).parents[2] / "detector"
    )
    landmark_xyz, landmark_desc, landmark_prior, stats = build_stdloc_detector_landmark_bank(
        model,
        sp_head,
        detector_dir=detector_dir,
        max_landmarks=args.max_landmarks,
        descriptor_source=args.descriptor_source,
        ply_loc_feature_weight=args.ply_loc_feature_weight,
        hybrid_residual_alpha_max=args.hybrid_residual_alpha_max,
        device=device,
        candidate_source="sampled",
    )

    dataset = CambridgeHybridDataset(
        scene_root=scene_root,
        cameras_json=train_args["cameras_json"],
        split="train",
        image_subdir=train_args.get("image_subdir", "processed"),
        image_height=train_args["image_height"],
        image_width=train_args["image_width"],
        fx=full_fx,
        fy=full_fy,
        cx=full_cx,
        cy=full_cy,
        max_frames=0,
    )
    visibility_renderer = None
    visibility_values = None
    if args.visibility_check == "rendered":
        from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

        if args.query_detector == "stdloc":
            visibility_renderer = FeatureFieldRenderer(
                image_height=train_args["image_height"],
                image_width=train_args["image_width"],
                fx=full_fx,
                fy=full_fy,
                cx=full_cx,
                cy=full_cy,
                max_channels_per_chunk=1,
                far_plane=10000.0,
                packed=False,
                rasterize_mode="antialiased",
            ).to(device)
        else:
            visibility_renderer = FeatureFieldRenderer(
                image_height=train_args["feature_height"],
                image_width=train_args["feature_width"],
                fx=feature_intr["fx"],
                fy=feature_intr["fy"],
                cx=feature_intr["cx"],
                cy=feature_intr["cy"],
                max_channels_per_chunk=1,
            ).to(device)
        visibility_values = torch.ones(model.get_xyz().shape[0], 1, device=device, dtype=torch.float32)
    ids = list(range(0, len(dataset), max(1, int(args.view_stride))))
    if args.max_views > 0:
        ids = ids[: int(args.max_views)]

    tp = torch.zeros(landmark_xyz.shape[0], device=device, dtype=torch.float32)
    fp = torch.zeros_like(tp)
    landmark_desc = F.normalize(landmark_desc.float(), p=2, dim=-1)
    collect_scene_pairs = bool(args.scene_match_pair_output_path)
    pair_sample_limit = max(0, int(args.scene_match_pair_sample_limit))
    train_pair_limit = int(round(pair_sample_limit * min(max(float(args.scene_match_pair_train_fraction), 0.0), 1.0)))
    rendered_pair_limit = pair_sample_limit - train_pair_limit
    pair_phase_limits = {"train": train_pair_limit, "rendered": rendered_pair_limit}
    pair_phase_counts = {"train": 0, "rendered": 0}
    pair_chunks: dict[str, list[torch.Tensor]] = {
        "query_desc": [],
        "landmark_desc": [],
        "cosine": [],
        "margin": [],
        "query_score": [],
        "landmark_prior": [],
        "label": [],
    }
    pair_sample_count = 0

    def should_collect_scene_match_pairs(phase: str) -> bool:
        if not collect_scene_pairs:
            return False
        phase = str(phase)
        if phase not in pair_phase_limits:
            raise ValueError(f"unsupported scene match pair phase: {phase}")
        phase_limit = int(pair_phase_limits[phase])
        return phase_limit > 0 and int(pair_phase_counts[phase]) < phase_limit

    def append_scene_match_pairs(
        *,
        phase: str,
        query_desc_all: torch.Tensor,
        q_ids: torch.Tensor,
        lm_ids: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        margins: torch.Tensor,
        query_scores_all: torch.Tensor | None = None,
    ) -> None:
        nonlocal pair_sample_count
        phase = str(phase)
        if phase not in pair_phase_limits:
            raise ValueError(f"unsupported scene match pair phase: {phase}")
        phase_limit = int(pair_phase_limits[phase])
        phase_count = int(pair_phase_counts[phase])
        if not should_collect_scene_match_pairs(phase):
            return
        remaining = phase_limit - phase_count
        labels = labels.bool().reshape(-1)
        scores = scores.float().reshape(-1)
        pos = torch.where(labels)[0]
        neg = torch.where(~labels)[0]
        selected = []
        pos_budget = min(int(pos.numel()), max(1, remaining // 2)) if pos.numel() else 0
        if pos_budget > 0:
            selected.append(pos[torch.argsort(scores[pos], descending=True)[:pos_budget]])
        neg_budget = min(int(neg.numel()), remaining - pos_budget)
        if neg_budget > 0:
            selected.append(neg[torch.argsort(scores[neg], descending=True)[:neg_budget]])
        if not selected:
            return
        keep = torch.cat(selected, dim=0)
        if keep.numel() > remaining:
            keep = keep[:remaining]
        pair_chunks["query_desc"].append(query_desc_all[q_ids[keep]].detach().cpu().half())
        pair_chunks["landmark_desc"].append(landmark_desc[lm_ids[keep]].detach().cpu().half())
        pair_chunks["cosine"].append(scores[keep].detach().cpu().float())
        pair_chunks["margin"].append(margins[keep].detach().cpu().float())
        if query_scores_all is None:
            query_scores = torch.ones_like(scores)
        else:
            query_scores = query_scores_all.to(device=scores.device, dtype=scores.dtype).reshape(-1)[q_ids]
        pair_chunks["query_score"].append(query_scores[keep].detach().cpu().float())
        pair_chunks["landmark_prior"].append(landmark_prior[lm_ids[keep]].detach().cpu().float())
        pair_chunks["label"].append(labels[keep].detach().cpu().float())
        added = int(keep.numel())
        pair_sample_count += added
        pair_phase_counts[phase] = phase_count + added

    def accumulate_query(
        *,
        phase: str,
        query_desc: torch.Tensor,
        query_yx: torch.Tensor,
        pose_w2c: torch.Tensor,
        K: torch.Tensor,
        query_scores: torch.Tensor | None = None,
        depth_map: torch.Tensor | None = None,
        alpha_map: torch.Tensor | None = None,
    ) -> None:
        if query_desc.numel() == 0:
            return
        raw_corr = F.normalize(query_desc.float(), p=2, dim=-1) @ landmark_desc.T
        corr = apply_match_prior(raw_corr.unsqueeze(0), landmark_prior, weight=args.locability_prior_weight)[0]
        k = min(max(1, int(args.topk)), int(corr.shape[-1]))
        _vals, lm_topk = torch.topk(corr, k=k, dim=-1)
        q_ids = torch.arange(lm_topk.shape[0], device=device).view(-1, 1).expand_as(lm_topk).reshape(-1)
        lm_ids = lm_topk.reshape(-1)
        top2 = torch.topk(corr, k=min(2, int(corr.shape[-1])), dim=-1).values
        if top2.shape[-1] == 2:
            margin_by_query = top2[:, 0] - top2[:, 1]
        else:
            margin_by_query = torch.zeros(corr.shape[0], device=device, dtype=corr.dtype)
        if should_collect_scene_match_pairs(phase):
            pair_labels = label_scene_match_pairs(
                query_yx=query_yx,
                landmark_xyz=landmark_xyz,
                q_ids=q_ids,
                lm_ids=lm_ids,
                pose_w2c=pose_w2c.to(device),
                K=K,
                reprojection_threshold_px=args.reprojection_threshold_px,
                depth_map=depth_map,
                alpha_map=alpha_map,
                depth_abs_tolerance=args.visibility_depth_abs_tolerance,
                depth_rel_tolerance=args.visibility_depth_rel_tolerance,
                alpha_threshold=args.visibility_alpha_threshold,
            )
            append_scene_match_pairs(
                phase=phase,
                query_desc_all=query_desc,
                q_ids=q_ids,
                lm_ids=lm_ids,
                scores=raw_corr[q_ids, lm_ids],
                labels=pair_labels,
                margins=margin_by_query[q_ids],
                query_scores_all=query_scores,
            )
        accumulate_matchability_counts(
            tp,
            fp,
            query_yx=query_yx,
            landmark_xyz=landmark_xyz,
            q_ids=q_ids,
            lm_ids=lm_ids,
            pose_w2c=pose_w2c.to(device),
            K=K,
            reprojection_threshold_px=args.reprojection_threshold_px,
            depth_map=depth_map,
            alpha_map=alpha_map,
            depth_abs_tolerance=args.visibility_depth_abs_tolerance,
            depth_rel_tolerance=args.visibility_depth_rel_tolerance,
            alpha_threshold=args.visibility_alpha_threshold,
        )

    for idx in tqdm(ids, desc=f"Calibrating {scene}", dynamic_ncols=True):
        item = dataset[int(idx)]
        resized_rgb = item["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
        teacher_rgb = resized_rgb
        if args.query_feature_source == "original":
            teacher_rgb = load_cambridge_rgb_no_resize(
                scene_root,
                train_args.get("image_subdir", "processed"),
                str(item["image_name"]),
                device,
            ).unsqueeze(0)
        with torch.no_grad():
            teacher_desc_raw_b, teacher_det_raw_b = teacher(superpoint_gray(teacher_rgb))
        teacher_desc_raw, teacher_desc, teacher_det = prepare_query_teacher_maps(
            teacher_desc_raw_b,
            teacher_det_raw_b,
            train_args["feature_height"],
            train_args["feature_width"],
        )
        if args.query_detector == "stdloc":
            if stdloc_detector is None:
                raise RuntimeError("query_detector=stdloc requires a detector")
            full_h, full_w = calibration_query_canvas_hw(resized_rgb, teacher_rgb)
            query_feature_map = F.normalize(upsample_feature_map(teacher_desc_raw, full_h, full_w), p=2, dim=0)
            keypoints, _scores = extract_stdloc_detector_keypoints(
                query_feature_map,
                stdloc_detector,
                max_keypoints=args.query_keypoints,
                nms_radius=args.nms_radius,
            )
            query_desc = sample_descriptors_bilinear(query_feature_map, keypoints)
            K = item["K"].to(device)
            query_for_label = keypoints + 0.5
        elif args.query_detector == "feedback":
            if feedback_detector is None:
                raise RuntimeError("query_detector=feedback requires a detector")
            if bool(args.feedback_detector_full_res):
                full_h, full_w = calibration_query_canvas_hw(resized_rgb, teacher_rgb)
                query_feature_map = F.normalize(upsample_feature_map(teacher_desc_raw, full_h, full_w), p=2, dim=0)
                keypoints, _scores = extract_stdloc_detector_keypoints(
                    query_feature_map,
                    feedback_detector,
                    max_keypoints=args.query_keypoints,
                    nms_radius=args.nms_radius,
                )
                query_desc = sample_descriptors_bilinear(query_feature_map, keypoints)
                K = item["K"].to(device)
                query_for_label = keypoints + 0.5
            else:
                keypoints, _scores = extract_stdloc_detector_keypoints(
                    teacher_desc,
                    feedback_detector,
                    max_keypoints=args.query_keypoints,
                    nms_radius=args.nms_radius,
                )
                query_desc = sample_descriptors_bilinear(teacher_desc, keypoints)
                K = item["feature_K"].to(device)
                query_for_label = keypoints
        else:
            keypoints, _scores = extract_keypoints_from_detector_logits(
                teacher_det,
                max_keypoints=args.query_keypoints,
                confidence_threshold=args.keypoint_threshold,
                nms_radius=args.nms_radius,
            )
            query_desc = sample_descriptors_bilinear(teacher_desc, keypoints)
            K = item["feature_K"].to(device)
            query_for_label = keypoints
        if query_desc.numel() == 0:
            continue
        depth_map = None
        alpha_map = None
        if visibility_renderer is not None and visibility_values is not None:
            visibility_renderer.K.copy_(K.to(device=device, dtype=visibility_renderer.K.dtype))
            vis_render = visibility_renderer.render_feature_values_batch(
                model,
                visibility_values,
                item["pose_w2c"].to(device=device, dtype=torch.float32).unsqueeze(0),
            )
            depth_map = vis_render["depth_map"][0]
            alpha_map = vis_render["alpha_map"][0]
        accumulate_query(
            phase="train",
            query_desc=query_desc,
            query_yx=query_for_label,
            pose_w2c=item["pose_w2c"].to(device),
            K=K,
            query_scores=_scores,
            depth_map=depth_map,
            alpha_map=alpha_map,
        )

    rendered_rehearsal_views = max(0, int(args.rendered_rehearsal_views))
    rendered_trans_deltas: list[float] = []
    rendered_rot_deltas: list[float] = []
    rendered_overlap_values: list[float] = []
    if rendered_rehearsal_views > 0:
        if args.query_detector not in {"stdloc", "feedback"}:
            raise ValueError("--rendered_rehearsal_views requires --query_detector stdloc or feedback")
        if args.query_detector == "stdloc" and stdloc_detector is None:
            raise RuntimeError("rendered rehearsal calibration requires a STDLoc detector")
        if args.query_detector == "feedback" and feedback_detector is None:
            raise RuntimeError("rendered rehearsal calibration requires a feedback detector")
        if args.query_detector == "feedback" and args.rendered_query_source != "feature_field":
            raise ValueError("feedback rendered rehearsal currently requires --rendered_query_source feature_field")
        from loc_gs.rendering.feature_renderer import FeatureFieldRenderer

        rendered_full_res = args.query_detector == "stdloc" or bool(args.feedback_detector_full_res)
        rendered_height = train_args["image_height"] if rendered_full_res else train_args["feature_height"]
        rendered_width = train_args["image_width"] if rendered_full_res else train_args["feature_width"]
        rendered_fx = full_fx if rendered_full_res else feature_intr["fx"]
        rendered_fy = full_fy if rendered_full_res else feature_intr["fy"]
        rendered_cx = full_cx if rendered_full_res else feature_intr["cx"]
        rendered_cy = full_cy if rendered_full_res else feature_intr["cy"]
        rendered_renderer = FeatureFieldRenderer(
            image_height=rendered_height,
            image_width=rendered_width,
            fx=rendered_fx,
            fy=rendered_fy,
            cx=rendered_cx,
            cy=rendered_cy,
            max_channels_per_chunk=32,
            far_plane=10000.0,
            packed=False,
            rasterize_mode="antialiased",
        ).to(device)
        rendered_K = torch.tensor(
            [[rendered_fx, 0.0, rendered_cx], [0.0, rendered_fy, rendered_cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        full_K = torch.tensor(
            [[full_fx, 0.0, full_cx], [0.0, full_fy, full_cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        xyz_probe = model.get_xyz().detach()
        xyz_probe = xyz_probe[:: max(1, xyz_probe.shape[0] // 4096)]
        for ridx in tqdm(
            range(rendered_rehearsal_views),
            desc=f"Calibrating rendered {scene}",
            dynamic_ncols=True,
        ):
            frame_idx = int(ids[ridx % len(ids)] if ids else ridx % len(dataset))
            item = dataset[frame_idx]
            base_pose = item["pose_w2c"].to(device=device, dtype=torch.float32).unsqueeze(0)
            loc_pose = sample_rehearsal_pose_batch(
                dataset=dataset,
                frame_indices=torch.tensor([frame_idx], device=device, dtype=torch.long),
                base_pose_w2c=base_pose,
                model=model,
                renderer=rendered_renderer,
                mode=args.rendered_rehearsal_pose_mode,
                trans_m=args.rendered_pose_noise_trans_m,
                rot_deg=args.rendered_pose_noise_rot_deg,
                pair_probability=args.rendered_rehearsal_pair_probability,
                interpolation_min=args.rendered_rehearsal_interpolation_min,
                interpolation_max=args.rendered_rehearsal_interpolation_max,
                pair_jitter_trans_m=args.rendered_pair_jitter_trans_m,
                pair_jitter_rot_deg=args.rendered_pair_jitter_rot_deg,
                min_overlap=args.rendered_view_pair_min_overlap,
                device=device,
            ).detach()
            trans_delta, rot_delta = pose_delta_trans_rot(base_pose, loc_pose)
            rendered_trans_deltas.append(float(trans_delta[0].detach().cpu()))
            rendered_rot_deltas.append(float(rot_delta[0].detach().cpu()))
            rendered_overlap_values.append(
                float(
                    projective_view_overlap(
                        xyz_probe,
                        base_pose[0],
                        loc_pose[0],
                        full_K,
                        train_args["image_height"],
                        train_args["image_width"],
                    )
                )
            )
            if args.rendered_query_source == "rendered_rgb_teacher":
                rgb_render = rendered_renderer.render_rgb(model, loc_pose[0])
                rendered_rgb = rgb_render["rgb"].unsqueeze(0).to(device=device, dtype=torch.float32)
                teacher_desc_raw_b, teacher_det_raw_b = teacher(superpoint_gray(rendered_rgb))
                teacher_desc_raw, _teacher_desc_grid, _teacher_det_grid = prepare_query_teacher_maps(
                    teacher_desc_raw_b,
                    teacher_det_raw_b,
                    train_args["feature_height"],
                    train_args["feature_width"],
                )
                query_feature_map = F.normalize(
                    upsample_feature_map(teacher_desc_raw, train_args["image_height"], train_args["image_width"]),
                    p=2,
                    dim=0,
                )
                depth_map = rgb_render["depth"]
                alpha_map = rgb_render["alpha"]
            else:
                rendered = render_hybrid_superpoint(
                    model,
                    sp_head,
                    rendered_renderer,
                    loc_pose,
                    descriptor_source=args.descriptor_source,
                    ply_loc_feature_weight=args.ply_loc_feature_weight,
                    hybrid_residual_alpha_max=args.hybrid_residual_alpha_max,
                )
                query_feature_map = F.normalize(rendered["descriptor"][0].float(), p=2, dim=0)
                depth_map = rendered["depth"][0]
                alpha_map = rendered["alpha"][0]
            rendered_detector = stdloc_detector if args.query_detector == "stdloc" else feedback_detector
            if rendered_detector is None:
                raise RuntimeError("rendered rehearsal detector is not loaded")
            keypoints, _scores = extract_stdloc_detector_keypoints(
                query_feature_map,
                rendered_detector,
                max_keypoints=args.query_keypoints,
                nms_radius=args.nms_radius,
            )
            query_desc = sample_descriptors_bilinear(query_feature_map, keypoints)
            accumulate_query(
                phase="rendered",
                query_desc=query_desc,
                query_yx=keypoints + (0.5 if rendered_full_res else 0.0),
                pose_w2c=loc_pose[0],
                K=rendered_K,
                query_scores=_scores,
                depth_map=depth_map,
                alpha_map=alpha_map,
            )

    matchability = matchability_from_counts(tp, fp, alpha=args.smoothing_alpha)
    fp_rate = fp / (tp + fp).clamp_min(1.0)
    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "mean": float(tensor.mean().item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }

    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(train_args.get("output_root", "output"))
        / "cache"
        / "calibrated_matchability"
        / "Cambridge_stdloc"
        / scene
        / "stdloc_bank.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "landmark_matchability": matchability.detach().cpu(),
        "landmark_tp_count": tp.detach().cpu(),
        "landmark_fp_count": fp.detach().cpu(),
        "landmark_fp_rate": fp_rate.detach().cpu(),
        "matchability_calibrator": {
            "type": "beta_smoothed_tp_fp",
            "smoothing_alpha": float(args.smoothing_alpha),
        },
        "metadata": {
            "scene": scene,
            "checkpoint": str(args.checkpoint),
            "views": len(ids),
            "rendered_rehearsal_views": rendered_rehearsal_views,
            "rendered_query_source": args.rendered_query_source,
            "rendered_rehearsal_pose_mode": args.rendered_rehearsal_pose_mode,
            "rendered_rehearsal_interpolation_min": float(args.rendered_rehearsal_interpolation_min),
            "rendered_rehearsal_interpolation_max": float(args.rendered_rehearsal_interpolation_max),
            "rendered_pose_noise_trans_m": float(args.rendered_pose_noise_trans_m),
            "rendered_pose_noise_rot_deg": float(args.rendered_pose_noise_rot_deg),
            "rendered_pose_delta_trans_m": _summary(rendered_trans_deltas),
            "rendered_pose_delta_rot_deg": _summary(rendered_rot_deltas),
            "rendered_base_overlap": _summary(rendered_overlap_values),
            "topk": int(args.topk),
            "reprojection_threshold_px": float(args.reprojection_threshold_px),
            "max_landmarks": int(args.max_landmarks),
            "query_detector": args.query_detector,
            "feedback_detector_path": args.feedback_detector_path,
            "query_feature_source": args.query_feature_source,
            "visibility_check": args.visibility_check,
            "visibility_alpha_threshold": float(args.visibility_alpha_threshold),
            "visibility_depth_abs_tolerance": float(args.visibility_depth_abs_tolerance),
            "visibility_depth_rel_tolerance": float(args.visibility_depth_rel_tolerance),
            "landmark_score_stats": stats,
        },
    }
    write_matchability_calibration(payload, output_path)
    if collect_scene_pairs:
        pair_output_path = Path(args.scene_match_pair_output_path)
        pair_output_path.parent.mkdir(parents=True, exist_ok=True)
        pair_payload = {
            key: torch.cat(chunks, dim=0) if chunks else torch.empty(0)
            for key, chunks in pair_chunks.items()
        }
        pair_payload["metadata"] = {
            "scene": scene,
            "checkpoint": str(args.checkpoint),
            "samples": int(pair_sample_count),
            "sample_limit": int(pair_sample_limit),
            "train_fraction": float(args.scene_match_pair_train_fraction),
            "phase_limits": {key: int(value) for key, value in pair_phase_limits.items()},
            "phase_counts": {key: int(value) for key, value in pair_phase_counts.items()},
            "topk": int(args.topk),
            "reprojection_threshold_px": float(args.reprojection_threshold_px),
            "source": "calibrate_landmark_matchability",
        }
        torch.save(pair_payload, pair_output_path)
        pair_sidecar = pair_output_path.with_suffix(".json")
        positives = pair_payload["label"].float().sum() if pair_payload["label"].numel() else torch.tensor(0.0)
        pair_sidecar.write_text(
            json.dumps(
                {
                    **pair_payload["metadata"],
                    "positives": float(positives.item()),
                    "positive_ratio": float(positives.item() / max(1, int(pair_sample_count))),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[calibrate] wrote {pair_output_path}")
    sidecar = output_path.with_suffix(".json")
    print(f"[calibrate] wrote {output_path}")
    print(f"[calibrate] wrote {sidecar}")


if __name__ == "__main__":
    main()
