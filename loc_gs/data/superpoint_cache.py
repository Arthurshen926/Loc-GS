from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
import uuid

import numpy as np
import torch
import torch.nn.functional as F


def superpoint_score_map_from_logits(detector_logits: torch.Tensor) -> np.ndarray:
    logits = detector_logits.detach().float().cpu()
    probs = F.softmax(logits, dim=0)
    heatmap = F.pixel_shuffle(probs[:64].unsqueeze(0), 8).squeeze(0).squeeze(0)
    return heatmap.numpy().astype(np.float32)


@dataclass
class SuperPointCacheEntry:
    descriptor: torch.Tensor
    detector_logits: torch.Tensor


class SuperPointTeacherCache:
    """Disk cache for Cambridge SuperPoint teacher outputs."""

    def __init__(
        self,
        output_root: str | Path,
        scene: str,
        split: str,
        dataset_name: str = "Cambridge_stdloc",
    ) -> None:
        root = Path(output_root)
        if root.name == "superpoint":
            self.root = root / dataset_name / scene / split
        else:
            self.root = root / "cache" / "superpoint" / dataset_name / scene / split
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stem(image_name: str) -> str:
        path = Path(image_name)
        stem = path.with_suffix("").as_posix().replace("/", "__")
        return stem

    def descriptor_path(self, image_name: str) -> Path:
        return self.root / f"{self._stem(image_name)}.pt"

    def score_path(self, image_name: str) -> Path:
        return self.root / f"{self._stem(image_name)}_score.npy"

    def metadata_path(self, image_name: str) -> Path:
        return self.root / f"{self._stem(image_name)}.npz"

    def load(self, image_name: str, map_location: Optional[torch.device | str] = None) -> Optional[SuperPointCacheEntry]:
        path = self.descriptor_path(image_name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location=map_location or "cpu")
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            return None
        if isinstance(payload, dict):
            descriptor = payload["descriptor"]
            detector = payload["detector_logits"]
        else:
            return None
        return SuperPointCacheEntry(descriptor=descriptor, detector_logits=detector)

    def save(
        self,
        image_name: str,
        descriptor: torch.Tensor,
        detector_logits: torch.Tensor,
        keypoints: Optional[torch.Tensor] = None,
        keypoint_descriptors: Optional[torch.Tensor] = None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor_path = self.descriptor_path(image_name)
        tmp_descriptor_path = descriptor_path.with_name(
            f"{descriptor_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        torch.save(
            {
                "descriptor": descriptor.detach().cpu(),
                "detector_logits": detector_logits.detach().cpu(),
            },
            tmp_descriptor_path,
        )
        os.replace(tmp_descriptor_path, descriptor_path)

        score_path = self.score_path(image_name)
        tmp_score_path = score_path.with_name(f"{score_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with tmp_score_path.open("wb") as f:
            np.save(f, superpoint_score_map_from_logits(detector_logits))
        os.replace(tmp_score_path, score_path)
        if keypoints is not None or keypoint_descriptors is not None:
            self.save_metadata(image_name, keypoints, keypoint_descriptors, descriptor_dim=descriptor.shape[0])

    def save_metadata(
        self,
        image_name: str,
        keypoints: Optional[torch.Tensor] = None,
        keypoint_descriptors: Optional[torch.Tensor] = None,
        descriptor_dim: int = 256,
    ) -> None:
        """Write keypoint metadata without touching dense teacher cache files."""
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.metadata_path(image_name)
        tmp_metadata_path = metadata_path.with_name(
            f"{metadata_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        with tmp_metadata_path.open("wb") as f:
            np.savez(
                f,
                keypoints=(
                    keypoints.detach().cpu().numpy()
                    if keypoints is not None
                    else np.zeros((0, 2), dtype=np.float32)
                ),
                descriptors=(
                    keypoint_descriptors.detach().cpu().numpy()
                    if keypoint_descriptors is not None
                    else np.zeros((0, int(descriptor_dim)), dtype=np.float32)
                ),
            )
        os.replace(tmp_metadata_path, metadata_path)
