from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def resolve_cambridge_masks_path(scene_root: str | Path, image_subdir: str = "processed") -> Path | None:
    """Resolve STDLoc Cambridge Mask2Former masks without assuming one image root layout."""
    root = Path(scene_root)
    image_subdir = str(image_subdir or "").strip("/")
    candidates: list[Path] = []
    if image_subdir and image_subdir != ".":
        candidates.append(root / image_subdir / "masks.pkl")
    candidates.append(root / "processed" / "masks.pkl")
    candidates.append(root / "masks.pkl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _as_bool_mask(mask: Any) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"Cambridge semantic mask must be HxW or 1xHxW, got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.bool, device="cpu")


def compose_cambridge_semantic_valid_mask(
    mask_tuple: tuple[Any, Any, Any] | list[Any],
    *,
    remove_sky: bool,
) -> torch.Tensor:
    """Compose STDLoc masks into valid reconstruction pixels.

    STDLoc preprocessing stores `(stuff_mask, sky_mask, undistort_mask)`.
    `stuff_mask` is false for Mask2Former thing/dynamic objects, `sky_mask`
    is false on sky, and `undistort_mask` is false on invalid undistorted borders.
    """
    if len(mask_tuple) < 3:
        raise ValueError(f"Expected STDLoc mask tuple length >= 3, got {len(mask_tuple)}")
    stuff_mask = _as_bool_mask(mask_tuple[0])
    sky_mask = _as_bool_mask(mask_tuple[1])
    distort_mask = _as_bool_mask(mask_tuple[2])
    valid = stuff_mask & distort_mask
    if remove_sky:
        valid = valid & sky_mask
    return valid.contiguous()


def resize_semantic_mask(mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    mask = _as_bool_mask(mask)
    if mask.shape == (int(height), int(width)):
        return mask
    resized = F.interpolate(
        mask[None, None].float(),
        size=(int(height), int(width)),
        mode="nearest",
    )
    return resized[0, 0] > 0.5


@dataclass
class CambridgeSemanticMaskStore:
    path: Path | None
    masks: dict[str, Any] = field(default_factory=dict)
    remove_sky: bool = True
    missing_image_names: set[str] = field(default_factory=set)

    @classmethod
    def from_scene(
        cls,
        scene_root: str | Path,
        *,
        image_subdir: str = "processed",
        remove_sky: bool = True,
        require: bool = False,
    ) -> "CambridgeSemanticMaskStore":
        path = resolve_cambridge_masks_path(scene_root, image_subdir=image_subdir)
        if path is None:
            if require:
                raise FileNotFoundError(
                    f"Cambridge semantic masks not found under {Path(scene_root)} "
                    f"for image_subdir={image_subdir!r}"
                )
            return cls(path=None, masks={}, remove_sky=remove_sky)
        with path.open("rb") as handle:
            masks = pickle.load(handle)
        if not isinstance(masks, dict):
            raise ValueError(f"Cambridge semantic masks must be a dict, got {type(masks)!r} from {path}")
        return cls(path=path, masks=masks, remove_sky=remove_sky)

    @property
    def loaded(self) -> bool:
        return self.path is not None and bool(self.masks)

    def valid_mask(self, image_name: str, fallback_hw: tuple[int, int] | None = None) -> torch.Tensor:
        entry = self.masks.get(image_name)
        if entry is None:
            self.missing_image_names.add(str(image_name))
            if fallback_hw is None:
                raise KeyError(f"Semantic mask missing for {image_name!r} and no fallback size was provided")
            height, width = fallback_hw
            return torch.ones(int(height), int(width), dtype=torch.bool)
        return compose_cambridge_semantic_valid_mask(entry, remove_sky=self.remove_sky)

    def audit(self) -> dict[str, object]:
        return {
            "loaded": self.loaded,
            "mask_path": str(self.path) if self.path is not None else None,
            "mask_count": len(self.masks),
            "remove_sky": bool(self.remove_sky),
            "missing_image_count": len(self.missing_image_names),
            "missing_image_names": sorted(self.missing_image_names)[:20],
        }
