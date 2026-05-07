#!/usr/bin/env python3
"""
Evaluate a trained SuperPoint hybrid feature field model.

Produces:
  1. Per-pixel cosine similarity maps (heatmaps)
  2. PCA visualizations comparing predicted vs GT descriptors
  3. Descriptor matching quality metrics
  4. Summary statistics (mean cosine, PSNR, per-channel stats)

Usage:
    python -m loc_gs.scripts.eval_superpoint \
        --config configs/superpoint_hybrid_room_0_v2.yaml \
        --checkpoint /root/Loc-GS/output/sp_gs/room0_hybrid_v2/checkpoints/best.pth \
        --output_dir /root/Loc-GS/output/sp_gs/room0_hybrid_v2/eval \
        --num_samples 20 \
        --device cuda:2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

from loc_gs.config import load_config
from loc_gs.scripts.train_feature_field import LocGSTrainer


def compute_per_pixel_cosine(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel cosine similarity between pred and gt.
    
    Args:
        pred: [C, H, W] predicted descriptors (L2-normed)
        gt: [C, H, W] ground truth descriptors (L2-normed)
    Returns:
        cosine_map: [H, W] per-pixel cosine similarity
    """
    return (pred * gt).sum(dim=0)  # [H, W]


def compute_pca_visualization(
    features: torch.Tensor, 
    pca_model: Optional[PCA] = None,
    n_components: int = 3,
) -> Tuple[np.ndarray, PCA]:
    """Apply PCA to feature maps for visualization.
    
    Args:
        features: [C, H, W] feature tensor
        pca_model: existing PCA model (fit on GT), or None to fit new
        n_components: number of PCA components
    Returns:
        rgb_vis: [H, W, 3] PCA visualization (0-255 uint8)
        pca_model: fitted PCA model
    """
    C, H, W = features.shape
    feat_flat = features.reshape(C, -1).T.cpu().numpy()  # [H*W, C]
    
    if pca_model is None:
        pca_model = PCA(n_components=n_components)
        pca_model.fit(feat_flat)
    
    projected = pca_model.transform(feat_flat)  # [H*W, 3]
    
    # Normalize to [0, 1] per channel
    for i in range(n_components):
        vmin, vmax = projected[:, i].min(), projected[:, i].max()
        if vmax - vmin > 1e-6:
            projected[:, i] = (projected[:, i] - vmin) / (vmax - vmin)
        else:
            projected[:, i] = 0.5
    
    rgb = (projected.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb, pca_model


def evaluate_descriptor_matching(
    pred_desc: torch.Tensor, 
    gt_desc: torch.Tensor,
    num_keypoints: int = 500,
) -> Dict[str, float]:
    """Evaluate descriptor matching quality using nearest-neighbor matching.
    
    Samples random keypoint locations, finds NN matches in predicted descriptors,
    and checks if they match the correct GT location.
    
    Args:
        pred_desc: [C, H, W] predicted descriptors
        gt_desc: [C, H, W] GT descriptors
        num_keypoints: number of keypoints to sample
    Returns:
        dict with matching metrics
    """
    C, H, W = pred_desc.shape
    device = pred_desc.device
    
    # Sample random keypoint locations
    total_pixels = H * W
    num_kp = min(num_keypoints, total_pixels)
    indices = torch.randperm(total_pixels, device=device)[:num_kp]
    rows = indices // W
    cols = indices % W
    
    # Extract GT descriptors at keypoint locations
    gt_kp = gt_desc[:, rows, cols]  # [C, K]
    
    # Flatten predicted descriptors
    pred_flat = pred_desc.reshape(C, -1)  # [C, H*W]
    
    # Find nearest neighbor in predicted space for each GT keypoint
    # Cosine similarity (both are L2-normed)
    sim_matrix = gt_kp.T @ pred_flat  # [K, H*W]
    nn_indices = sim_matrix.argmax(dim=1)  # [K]
    
    # Check how many match the correct pixel (exact match)
    exact_match = (nn_indices == indices).float().mean().item()
    
    # Check within radius (3 pixels)
    nn_rows = nn_indices // W
    nn_cols = nn_indices % W
    dist = ((nn_rows - rows).float() ** 2 + (nn_cols - cols).float() ** 2).sqrt()
    match_r3 = (dist <= 3.0).float().mean().item()
    match_r5 = (dist <= 5.0).float().mean().item()
    
    # Mean NN cosine similarity
    nn_cos = sim_matrix.max(dim=1).values.mean().item()
    
    return {
        "exact_match_rate": exact_match,
        "match_r3": match_r3,
        "match_r5": match_r5,
        "mean_nn_cosine": nn_cos,
        "mean_dist": dist.mean().item(),
        "median_dist": dist.median().item(),
    }


@torch.no_grad()
def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 20,
    device: str = "cuda:0",
):
    """Run full evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config and create trainer (which loads model)
    config = load_config(config_path)
    config.device = device
    
    trainer = LocGSTrainer(config)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "sp_output_head_state_dict" in ckpt:
        trainer.sp_output_head.load_state_dict(ckpt["sp_output_head_state_dict"])
    if "sharpener_state_dict" in ckpt:
        trainer.sharpener.load_state_dict(ckpt["sharpener_state_dict"])
    if (
        "sp_locability_adapter_state_dict" in ckpt
        and getattr(trainer, "sp_locability_adapter", None) is not None
    ):
        trainer.sp_locability_adapter.load_state_dict(
            ckpt["sp_locability_adapter_state_dict"], strict=False
        )
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"best_cosine={ckpt.get('best_cosine', '?')}")
    
    trainer.model.eval()
    trainer.sp_output_head.eval()
    trainer.sharpener.eval()
    if getattr(trainer, "sp_locability_adapter", None) is not None:
        trainer.sp_locability_adapter.eval()
    
    # Collect metrics
    all_cosine = []
    all_mse = []
    all_matching = []
    
    # Use validation loader
    val_iter = iter(trainer.val_loader)
    
    pca_model = None  # Will be fit on first GT sample
    
    for i in range(min(num_samples, len(trainer.val_loader))):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        
        gt_features = batch["teacher_features"].to(device).float()
        pose_w2c = batch["pose_w2c"].to(device).float()
        
        # Render
        result = trainer.renderer.render_features_batch(trainer.model, pose_w2c)
        rendered_compact = result["feature_map"]
        rendered_compact = trainer.sharpener(rendered_compact)
        
        # Hybrid decode
        if trainer._is_hybrid:
            from loc_gs.models.hybrid_gaussian import unproject_depth_to_positions
            depth_map = result["depth_map"].float()
            position_map = unproject_depth_to_positions(
                depth_map, pose_w2c.float(), trainer.renderer.K.float(),
                depth_map.shape[1], depth_map.shape[2],
            )
            position_map = trainer._normalize_positions(position_map)
            rendered_compact = trainer.model.decode_screen_space(
                rendered_compact.float(), position_map,
            )
        
        locability_map = trainer._render_locability_map(pose_w2c.float())
        rendered_compact = trainer._apply_locability_to_sp_features(
            rendered_compact,
            locability_map,
        )

        # SP output head
        sp_out = trainer.sp_output_head(rendered_compact)
        pred_desc = sp_out["descriptor"]  # [B, 256, H, W]
        pred_det = sp_out["detector"]     # [B, 65, H, W]
        
        gt_desc = gt_features
        if gt_desc.shape[-2:] != pred_desc.shape[-2:]:
            gt_desc = F.interpolate(
                gt_desc, size=pred_desc.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        gt_desc = F.normalize(gt_desc.float(), p=2, dim=1)
        
        # Process each sample in batch
        B = pred_desc.shape[0]
        for b in range(B):
            sample_idx = i * B + b
            if sample_idx >= num_samples:
                break
            
            p = pred_desc[b]  # [256, H, W]
            g = gt_desc[b]    # [256, H, W]
            
            # Per-pixel cosine
            cos_map = compute_per_pixel_cosine(p, g)  # [H, W]
            mean_cos = cos_map.mean().item()
            all_cosine.append(mean_cos)
            
            # MSE
            mse = F.mse_loss(p, g).item()
            all_mse.append(mse)
            
            # Descriptor matching
            match_metrics = evaluate_descriptor_matching(p, g)
            all_matching.append(match_metrics)
            
            # Generate visualizations for first 10 samples
            if sample_idx < 10:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(
                    f"Sample {sample_idx} | Cosine={mean_cos:.4f} | "
                    f"Match@3px={match_metrics['match_r3']:.3f} | "
                    f"NN-cos={match_metrics['mean_nn_cosine']:.4f}",
                    fontsize=14,
                )
                
                # 1. Cosine similarity heatmap
                cos_np = cos_map.cpu().numpy()
                im = axes[0, 0].imshow(cos_np, cmap="RdYlGn", vmin=0, vmax=1)
                axes[0, 0].set_title(f"Cosine Similarity (mean={mean_cos:.4f})")
                plt.colorbar(im, ax=axes[0, 0])
                
                # 2. PCA of GT descriptors
                gt_pca_vis, pca_model = compute_pca_visualization(g, pca_model)
                axes[0, 1].imshow(gt_pca_vis)
                axes[0, 1].set_title("GT Descriptor (PCA)")
                
                # 3. PCA of predicted descriptors (using same PCA)
                pred_pca_vis, _ = compute_pca_visualization(p, pca_model)
                axes[0, 2].imshow(pred_pca_vis)
                axes[0, 2].set_title("Predicted Descriptor (PCA)")
                
                # 4. Cosine error map (1 - cosine)
                error_np = 1.0 - cos_np
                im2 = axes[1, 0].imshow(error_np, cmap="hot", vmin=0, vmax=1)
                axes[1, 0].set_title("Error (1 - cosine)")
                plt.colorbar(im2, ax=axes[1, 0])
                
                # 5. GT detector heatmap
                gt_det_batch = batch.get("detector_features")
                if gt_det_batch is not None:
                    gt_det = gt_det_batch[b].float()
                    gt_det_prob = F.softmax(gt_det, dim=0)  # [65, H, W]
                    # Sum over 64 cell bins (exclude dustbin=64)
                    gt_kp_prob = 1.0 - gt_det_prob[-1]  # keypoint probability
                    axes[1, 1].imshow(gt_kp_prob.cpu().numpy(), cmap="hot", vmin=0, vmax=0.5)
                    axes[1, 1].set_title("GT Keypoint Prob")
                else:
                    axes[1, 1].set_title("No GT detector")
                
                # 6. Predicted detector heatmap
                pred_det_prob = F.softmax(pred_det[b], dim=0)  # [65, H, W]
                pred_kp_prob = 1.0 - pred_det_prob[-1]
                axes[1, 2].imshow(pred_kp_prob.cpu().numpy(), cmap="hot", vmin=0, vmax=0.5)
                axes[1, 2].set_title("Predicted Keypoint Prob")
                
                for ax in axes.flat:
                    ax.axis("off")
                
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"sample_{sample_idx:03d}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
                print(f"  Saved sample_{sample_idx:03d}.png")
    
    # Compute summary statistics
    mean_cosine = np.mean(all_cosine)
    std_cosine = np.std(all_cosine)
    mean_mse = np.mean(all_mse)
    psnr = -10.0 * np.log10(mean_mse + 1e-8)
    
    # Average matching metrics
    avg_matching = {}
    for key in all_matching[0]:
        avg_matching[key] = np.mean([m[key] for m in all_matching])
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Samples evaluated: {len(all_cosine)}")
    print(f"Mean cosine similarity: {mean_cosine:.4f} (+/- {std_cosine:.4f})")
    print(f"MSE: {mean_mse:.6f}")
    print(f"PSNR: {psnr:.2f} dB")
    print()
    print("Descriptor Matching Quality:")
    print(f"  Exact match rate:    {avg_matching['exact_match_rate']:.4f}")
    print(f"  Match within 3px:    {avg_matching['match_r3']:.4f}")
    print(f"  Match within 5px:    {avg_matching['match_r5']:.4f}")
    print(f"  Mean NN cosine:      {avg_matching['mean_nn_cosine']:.4f}")
    print(f"  Mean match distance: {avg_matching['mean_dist']:.2f} px")
    print(f"  Median match dist:   {avg_matching['median_dist']:.2f} px")
    print("=" * 60)
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("SuperPoint Hybrid Feature Field Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Samples: {len(all_cosine)}\n\n")
        f.write(f"Mean cosine similarity: {mean_cosine:.4f} (+/- {std_cosine:.4f})\n")
        f.write(f"MSE: {mean_mse:.6f}\n")
        f.write(f"PSNR: {psnr:.2f} dB\n\n")
        f.write("Descriptor Matching:\n")
        for k, v in avg_matching.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nPer-sample cosine:\n")
        for i, c in enumerate(all_cosine):
            f.write(f"  Sample {i}: {c:.4f}\n")
    
    print(f"\nSummary saved to {summary_path}")
    
    # Plot cosine distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_cosine, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(mean_cosine, color="red", linestyle="--", label=f"Mean={mean_cosine:.4f}")
    ax.set_xlabel("Mean Per-Image Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Cosine Similarity Across Validation Images")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cosine_distribution.png"), dpi=150)
    plt.close()
    
    return {
        "mean_cosine": mean_cosine,
        "std_cosine": std_cosine,
        "psnr": psnr,
        "matching": avg_matching,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SuperPoint hybrid feature field")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", required=True, help="Directory for evaluation outputs")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()
    
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )
