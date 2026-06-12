import pickle
import struct
from pathlib import Path

import numpy as np
import torch

from loc_gs.sparse.landmarks import (
    CacheLandmarkResolver,
    load_base_gaussian_ids,
    load_gaussian_landmark_map,
    load_sampled_gaussian_ids,
)


def _write_binary_ply(path: Path) -> Path:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float loc_0\n"
        "property float loc_1\n"
        "end_header\n"
    )
    rows = [
        (1.0, 2.0, 3.0, 0.1, 0.2),
        (4.0, 5.0, 6.0, 0.3, 0.4),
        (7.0, 8.0, 9.0, 0.5, 0.6),
    ]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<5f", *row))
    return path


def test_gaussian_landmark_map_reads_binary_ply_xyz_and_descriptors(tmp_path: Path):
    path = _write_binary_ply(tmp_path / "point_cloud.ply")

    landmark_map = load_gaussian_landmark_map(path, load_descriptors=True)

    assert landmark_map.vertex_count == 3
    np.testing.assert_allclose(landmark_map.lookup_xyz([2, 0]), np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(landmark_map.lookup_descriptors([1]), np.array([[0.3, 0.4]], dtype=np.float32))
    assert landmark_map.descriptor_dim == 2


def test_cache_landmark_resolver_maps_cache_rows_to_gaussian_xyz(tmp_path: Path):
    ply = _write_binary_ply(tmp_path / "point_cloud.ply")
    pair_cache = tmp_path / "pairs.pt"
    torch.save({"base_gaussian_id": torch.tensor([2, 0, 1], dtype=torch.int64)}, pair_cache)

    landmark_map = load_gaussian_landmark_map(ply)
    resolver = CacheLandmarkResolver.from_pair_cache(pair_cache, landmark_map)

    assert load_base_gaussian_ids(pair_cache).tolist() == [2, 0, 1]
    np.testing.assert_array_equal(resolver.resolve_gaussian_ids([[0, 2], [1, -1]]), np.array([[2, 1], [0, -1]]))
    np.testing.assert_allclose(
        resolver.resolve_xyz([[0, 2], [1, -1]]),
        np.array(
            [
                [[7.0, 8.0, 9.0], [4.0, 5.0, 6.0]],
                [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]],
            ],
            dtype=np.float64,
        ),
        equal_nan=True,
    )


def test_sampled_gaussian_ids_loader_handles_pickle_tensors(tmp_path: Path):
    path = tmp_path / "sampled_idx.pkl"
    with path.open("wb") as handle:
        pickle.dump(torch.tensor([5, 3, 1], dtype=torch.int64), handle)

    sampled = load_sampled_gaussian_ids(path)

    np.testing.assert_array_equal(sampled, np.array([5, 3, 1], dtype=np.int64))
