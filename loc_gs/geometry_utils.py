from __future__ import annotations

from pathlib import Path

from plyfile import PlyData


def infer_ply_uses_2dgs(ply_path: str | Path) -> bool:
    """Infer whether a Gaussian PLY stores 2DGS surfels or 3DGS ellipsoids."""
    ply = PlyData.read(str(ply_path))
    vertex = ply.elements[0]
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    if len(scale_names) == 2:
        return True
    if len(scale_names) == 3:
        return False
    raise ValueError(
        f"Unable to infer Gaussian type from {ply_path}: expected 2 or 3 scale_* properties, got {len(scale_names)}"
    )


def resolve_use_2dgs(config, ply_path: str | Path | None = None) -> bool:
    """Resolve rasterization mode, preferring the actual PLY over config flags."""
    path = ply_path or getattr(config, "ply_path", "")
    if path:
        return infer_ply_uses_2dgs(path)
    return bool(getattr(config, "use_2dgs", False))
