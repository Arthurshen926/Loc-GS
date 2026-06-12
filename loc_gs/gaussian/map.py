from __future__ import annotations

from pathlib import Path

from loc_gs.sparse.landmarks import GaussianLandmarkMap, load_gaussian_landmark_map

GaussianMap = GaussianLandmarkMap


def load_gaussian_map(
    ply_path: str | Path,
    *,
    load_descriptors: bool = False,
    descriptor_prefix: str = "loc_",
) -> GaussianMap:
    return load_gaussian_landmark_map(
        ply_path,
        load_descriptors=bool(load_descriptors),
        descriptor_prefix=str(descriptor_prefix),
    )
