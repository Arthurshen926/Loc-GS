from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FeatureCacheEntry:
    image_name: str
    pipeline: str
    keypoints_xy: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: tuple[int, int]


@dataclass
class MatchCacheEntry:
    image_a: str
    image_b: str
    pipeline: str
    kpts_a_xy: np.ndarray
    kpts_b_xy: np.ndarray
    scores: np.ndarray
    geom_inlier_mask: np.ndarray
    stats: dict


class ExternalMatchCache:
    """Cache DIM/external sparse features and pair matches under output/cache."""

    def __init__(
        self,
        output_root: str | Path,
        scene: str,
        pipeline: str,
        split: str = "train",
        dataset_name: str = "Cambridge_stdloc",
    ) -> None:
        root = Path(output_root)
        self.scene = scene
        self.pipeline = str(pipeline).strip().lower()
        self.split = split
        if root.name in {"features", "matches"}:
            cache_root = root.parent
        elif root.name == "cache":
            cache_root = root
        else:
            cache_root = root / "cache"
        self.feature_root = cache_root / "features" / dataset_name / scene / split / self.pipeline
        self.match_root = cache_root / "matches" / dataset_name / scene / self.pipeline
        self.feature_root.mkdir(parents=True, exist_ok=True)
        self.match_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stem(image_name: str) -> str:
        return Path(image_name).with_suffix("").as_posix().replace("/", "__")

    def _pair_id(self, image_a: str, image_b: str) -> str:
        return f"{self._stem(image_a)}__TO__{self._stem(image_b)}"

    def feature_path(self, image_name: str) -> Path:
        return self.feature_root / f"{self._stem(image_name)}.npz"

    def match_path(self, image_a: str, image_b: str) -> Path:
        return self.match_root / f"{self._pair_id(image_a, image_b)}.npz"

    def save_features(
        self,
        image_name: str,
        keypoints_xy: np.ndarray,
        descriptors: np.ndarray,
        scores: np.ndarray,
        image_size: tuple[int, int],
    ) -> None:
        np.savez(
            self.feature_path(image_name),
            image_name=image_name,
            pipeline=self.pipeline,
            keypoints_xy=np.asarray(keypoints_xy, dtype=np.float32),
            descriptors=np.asarray(descriptors, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            image_size=np.asarray(image_size, dtype=np.int32),
            descriptor_dim=np.asarray([descriptors.shape[1] if descriptors.ndim == 2 else 0], dtype=np.int32),
        )

    def load_features(self, image_name: str) -> Optional[FeatureCacheEntry]:
        path = self.feature_path(image_name)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        image_size = tuple(int(v) for v in data["image_size"].tolist())
        return FeatureCacheEntry(
            image_name=str(data["image_name"]),
            pipeline=str(data["pipeline"]),
            keypoints_xy=data["keypoints_xy"].astype(np.float32),
            descriptors=data["descriptors"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            image_size=(image_size[0], image_size[1]),
        )

    def save_matches(
        self,
        image_a: str,
        image_b: str,
        kpts_a_xy: np.ndarray,
        kpts_b_xy: np.ndarray,
        scores: np.ndarray,
        geom_inlier_mask: np.ndarray,
        stats: Optional[dict] = None,
    ) -> None:
        np.savez(
            self.match_path(image_a, image_b),
            image_a=image_a,
            image_b=image_b,
            pipeline=self.pipeline,
            kpts_a_xy=np.asarray(kpts_a_xy, dtype=np.float32),
            kpts_b_xy=np.asarray(kpts_b_xy, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            geom_inlier_mask=np.asarray(geom_inlier_mask, dtype=np.bool_),
            stats_json=json.dumps(stats or {}),
        )

    def load_matches(self, image_a: str, image_b: str) -> Optional[MatchCacheEntry]:
        path = self.match_path(image_a, image_b)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        return MatchCacheEntry(
            image_a=str(data["image_a"]),
            image_b=str(data["image_b"]),
            pipeline=str(data["pipeline"]),
            kpts_a_xy=data["kpts_a_xy"].astype(np.float32),
            kpts_b_xy=data["kpts_b_xy"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            geom_inlier_mask=data["geom_inlier_mask"].astype(np.bool_),
            stats=json.loads(str(data["stats_json"])),
        )
