#!/usr/bin/env python3
"""
Extract and save SuperPoint features for all training images in a scene.

Outputs per-frame:
    descriptor/rgb_{idx}.pt  — [256, H/8, W/8] float16, L2-normalised dense descriptors
    detector/rgb_{idx}.pt    — [65, H/8, W/8]  float16, detector logits (pre-softmax)

Also computes PCA statistics over descriptors for visualization.

Usage:
    python -m loc_gs.scripts.extract_superpoint_features \
        --scene room_0 \
        --image_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/rgb/ \
        --output_dir output/superpoint_features/room_0/Sequence_1 \
        --weights third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth \
        --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from loc_gs.data.benchmark_paths import extract_feature_frame_index

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---- SuperPoint model (self-contained) ------------------------------------

class SuperPointNet(nn.Module):
    """Standard SuperPoint architecture (DeTone et al. 2018).

    VGG-style shared encoder with separate detector (65ch) and descriptor (256d) heads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # Shared encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector head: outputs [B, 65, H/8, W/8]
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor head: outputs [B, 256, H/8, W/8]
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: [B, 1, H, W] grayscale images in [0, 1].

        Returns:
            descriptors: [B, 256, H/8, W/8] L2-normalised dense descriptors.
            detector_logits: [B, 65, H/8, W/8] raw logits (pre-softmax).
        """
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector head
        cPa = self.relu(self.convPa(x))
        detector_logits = self.convPb(cPa)  # [B, 65, H/8, W/8]

        # Descriptor head
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)  # [B, 256, H/8, W/8]
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return descriptors, detector_logits


# ---- helpers ---------------------------------------------------------------

def _collect_image_paths(image_dir: str) -> tuple[list[Path], str]:
    paths = [
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    indexed: list[tuple[int, Path]] = []
    for path in paths:
        try:
            indexed.append((extract_feature_frame_index(path), path))
        except ValueError:
            indexed = []
            break
    if indexed:
        indexed.sort(key=lambda item: item[0])
        return [path for _, path in indexed], "numeric"
    return sorted(paths), "lexicographic"


def _load_and_preprocess(
    paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    """Load images, convert to grayscale, normalise to [0,1] → [B, 1, H, W]."""
    gray_transform = transforms.Grayscale(num_output_channels=1)
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = transforms.ToTensor()(img)  # [3, H, W] in [0, 1]
        t = gray_transform(t)  # [1, H, W]
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _compute_pca_stats(
    all_features: list[torch.Tensor],
    n_components: int = 64,
) -> dict[str, torch.Tensor]:
    pixels = torch.cat([f.reshape(f.shape[0], -1).T for f in all_features], dim=0)
    mean = pixels.mean(dim=0)
    std = pixels.std(dim=0).clamp(min=1e-6)
    centered = pixels - mean
    k = min(n_components, centered.shape[0], centered.shape[1])
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:k]
    return {"mean": mean, "std": std, "components_64": components}


# ---- main extraction -------------------------------------------------------

@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"[SuperPoint] Loading model weights from {args.weights}")
    model = SuperPointNet().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    # Filter keys: the official weights may have extra keys (e.g. from
    # the full pipeline). Load with strict=False.
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Collect images
    image_paths, image_sort_mode = _collect_image_paths(args.image_dir)
    print(f"[SuperPoint] Found {len(image_paths)} images in {args.image_dir}")
    print(f"[SuperPoint] Image ordering: {image_sort_mode}")

    # Probe resolution
    probe_img = Image.open(image_paths[0])
    orig_w, orig_h = probe_img.size
    # SuperPoint expects H, W divisible by 8
    target_h = (orig_h // 8) * 8
    target_w = (orig_w // 8) * 8
    feat_h, feat_w = target_h // 8, target_w // 8
    print(f"[SuperPoint] Input resolution: {orig_h}x{orig_w} -> {target_h}x{target_w}")
    print(f"[SuperPoint] Feature grid: {feat_h}x{feat_w}")

    # Prepare output dirs
    for sd in ["descriptor", "detector"]:
        os.makedirs(os.path.join(args.output_dir, sd), exist_ok=True)

    pca_accumulator: list[torch.Tensor] = []
    frame_manifest: list[dict[str, object]] = []
    total_bytes: int = 0
    t0 = time.time()

    # Process in batches
    n = len(image_paths)
    for start in tqdm(range(0, n, args.batch_size), desc="Extracting SuperPoint features"):
        batch_paths = image_paths[start : start + args.batch_size]
        imgs = _load_and_preprocess(batch_paths, device)  # [B, 1, H, W]

        # Resize to target resolution if needed
        if imgs.shape[2] != target_h or imgs.shape[3] != target_w:
            imgs = F.interpolate(imgs, size=(target_h, target_w), mode="bilinear", align_corners=False)

        with torch.cuda.amp.autocast(enabled=args.amp):
            descriptors, detector_logits = model(imgs)
            # descriptors: [B, 256, H/8, W/8], detector_logits: [B, 65, H/8, W/8]

        B = descriptors.shape[0]
        for i in range(B):
            source_path = batch_paths[i]
            source_rank = start + i
            try:
                frame_idx = extract_feature_frame_index(source_path)
            except ValueError:
                frame_idx = source_rank
            stem = f"rgb_{frame_idx}"
            frame_manifest.append({
                "source_rank": source_rank,
                "frame_idx": frame_idx,
                "source_file": source_path.name,
                "saved_stem": stem,
            })

            # Descriptor: float16
            desc = descriptors[i].cpu().half()
            desc_path = os.path.join(args.output_dir, "descriptor", f"{stem}.pt")
            torch.save(desc, desc_path)
            total_bytes += desc.nelement() * desc.element_size()

            # Detector logits: float16
            det = detector_logits[i].cpu().half()
            det_path = os.path.join(args.output_dir, "detector", f"{stem}.pt")
            torch.save(det, det_path)
            total_bytes += det.nelement() * det.element_size()

            # Accumulate descriptors for PCA
            pca_accumulator.append(desc.float())

    # PCA statistics
    print("[SuperPoint] Computing PCA statistics ...")
    pca_stats = _compute_pca_stats(pca_accumulator, n_components=64)
    pca_path = os.path.join(args.output_dir, "pca_stats.pt")
    torch.save(pca_stats, pca_path)

    manifest_path = Path(args.output_dir) / "frame_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "scene": args.scene,
            "image_dir": str(Path(args.image_dir).resolve()),
            "image_sort_mode": image_sort_mode,
            "num_frames": len(frame_manifest),
            "feature_type": "superpoint",
            "descriptor_dim": 256,
            "detector_dim": 65,
            "feature_height": feat_h,
            "feature_width": feat_w,
            "frames": frame_manifest,
        }, indent=2),
        encoding="utf-8",
    )

    # Summary
    elapsed = time.time() - t0
    disk_mb = total_bytes / (1024 * 1024)
    print("=" * 60)
    print(f"  Scene        : {args.scene}")
    print(f"  Frames       : {n}")
    print(f"  Descriptor   : 256 x {feat_h}x{feat_w}")
    print(f"  Detector     : 65 x {feat_h}x{feat_w}")
    print(f"  Disk usage   : {disk_mb:.1f} MB (float16)")
    print(f"  PCA saved    : {pca_path}")
    print(f"  Manifest     : {manifest_path}")
    print(f"  Time         : {elapsed:.1f}s ({elapsed / n:.2f}s/frame)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SuperPoint features for all images in a scene."
    )
    parser.add_argument("--scene", type=str, default="room_0")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--weights", type=str,
        default="third_party/stdloc/encoders/sp_encoder/weights/superpoint_v1.pth",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
