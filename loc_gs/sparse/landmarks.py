from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_PLY_DTYPE_MAP: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class GaussianLandmarkMap:
    source_path: str
    positions_xyz: np.ndarray
    descriptors: np.ndarray | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.positions_xyz.shape[0])

    @property
    def descriptor_dim(self) -> int:
        if self.descriptors is None:
            return 0
        return int(self.descriptors.shape[1])

    def lookup_xyz(self, gaussian_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(gaussian_ids, dtype=np.int64)
        result = np.full(ids.shape + (3,), np.nan, dtype=np.float64)
        valid = (ids >= 0) & (ids < self.vertex_count)
        if np.any(valid):
            result[valid] = self.positions_xyz[ids[valid]]
        return result

    def lookup_descriptors(self, gaussian_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        if self.descriptors is None:
            raise ValueError("landmark descriptors were not loaded")
        ids = np.asarray(gaussian_ids, dtype=np.int64)
        result = np.full(ids.shape + (self.descriptor_dim,), np.nan, dtype=np.float32)
        valid = (ids >= 0) & (ids < self.vertex_count)
        if np.any(valid):
            result[valid] = self.descriptors[ids[valid]]
        return result


@dataclass(frozen=True)
class PlyHeader:
    fmt: str
    vertex_count: int
    properties: tuple[tuple[str, str], ...]
    header_bytes: int

    def dtype(self) -> np.dtype:
        fields: list[tuple[str, str]] = []
        for prop_type, prop_name in self.properties:
            if prop_type not in _PLY_DTYPE_MAP:
                raise ValueError(f"unsupported PLY property type: {prop_type}")
            fields.append((prop_name, _PLY_DTYPE_MAP[prop_type]))
        return np.dtype(fields)


@dataclass(frozen=True)
class CacheLandmarkResolver:
    base_gaussian_ids: np.ndarray
    landmark_map: GaussianLandmarkMap

    @classmethod
    def from_pair_cache(cls, pair_cache_path: str | Path, landmark_map: GaussianLandmarkMap) -> "CacheLandmarkResolver":
        return cls(base_gaussian_ids=load_base_gaussian_ids(pair_cache_path), landmark_map=landmark_map)

    def resolve_gaussian_ids(self, cache_landmark_ids: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        ids = np.asarray(cache_landmark_ids, dtype=np.int64)
        result = np.full(ids.shape, -1, dtype=np.int64)
        valid = (ids >= 0) & (ids < int(self.base_gaussian_ids.shape[0]))
        if np.any(valid):
            result[valid] = self.base_gaussian_ids[ids[valid]]
        return result

    def resolve_xyz(self, cache_landmark_ids: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
        return self.landmark_map.lookup_xyz(self.resolve_gaussian_ids(cache_landmark_ids))


def _read_ply_header(path: Path) -> PlyHeader:
    properties: list[tuple[str, str]] = []
    fmt: str | None = None
    vertex_count: int | None = None
    in_vertex = False
    with path.open("rb") as handle:
        first = handle.readline()
        if first != b"ply\n":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError(f"PLY header is missing end_header: {path}")
            line = line_bytes.decode("ascii", errors="strict").strip()
            if line == "end_header":
                return PlyHeader(
                    fmt=str(fmt or ""),
                    vertex_count=int(vertex_count if vertex_count is not None else -1),
                    properties=tuple(properties),
                    header_bytes=int(handle.tell()),
                )
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                fmt = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex and len(parts) == 3:
                properties.append((parts[1], parts[2]))


def _descriptor_property_names(properties: Sequence[tuple[str, str]], *, prefix: str) -> list[str]:
    names = [name for _prop_type, name in properties if name.startswith(prefix)]

    def key(name: str) -> tuple[int, str]:
        suffix = name[len(prefix) :]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(names, key=key)


def load_gaussian_landmark_map(
    ply_path: str | Path,
    *,
    load_descriptors: bool = False,
    descriptor_prefix: str = "loc_",
) -> GaussianLandmarkMap:
    path = Path(ply_path)
    header = _read_ply_header(path)
    if header.fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format for internal landmark map: {header.fmt}")
    if header.vertex_count < 0:
        raise ValueError(f"PLY file is missing vertex count: {path}")
    prop_names = {name for _prop_type, name in header.properties}
    for required in ("x", "y", "z"):
        if required not in prop_names:
            raise ValueError(f"PLY file is missing required vertex property: {required}")

    dtype = header.dtype()
    data = np.memmap(path, mode="r", dtype=dtype, offset=header.header_bytes, shape=(header.vertex_count,))
    xyz = np.stack([np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])], axis=1).astype(np.float64)
    descriptors = None
    if load_descriptors:
        descriptor_names = _descriptor_property_names(header.properties, prefix=descriptor_prefix)
        if not descriptor_names:
            raise ValueError(f"PLY file has no descriptor properties with prefix {descriptor_prefix!r}")
        descriptors = np.stack([np.asarray(data[name]) for name in descriptor_names], axis=1).astype(np.float32)
    return GaussianLandmarkMap(source_path=str(path), positions_xyz=xyz, descriptors=descriptors)


def _load_torch_pt_mapping(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is available in the project env.
        raise RuntimeError("torch is required to read cached .pt landmark artifacts") from exc
    return torch.load(path, map_location="cpu")


def _to_numpy_int64(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64)


def load_base_gaussian_ids(pair_cache_path: str | Path) -> np.ndarray:
    path = Path(pair_cache_path)
    payload = _load_torch_pt_mapping(path)
    if not isinstance(payload, dict) or "base_gaussian_id" not in payload:
        raise ValueError(f"pair cache is missing base_gaussian_id: {path}")
    return _to_numpy_int64(payload["base_gaussian_id"])


def load_sampled_gaussian_ids(sampled_idx_path: str | Path) -> np.ndarray:
    path = Path(sampled_idx_path)
    with path.open("rb") as handle:
        sampled = pickle.load(handle)
    return _to_numpy_int64(sampled)
