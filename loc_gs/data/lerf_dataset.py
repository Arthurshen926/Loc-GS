"""LERF dataset loader for Loc-GS text grounding evaluation.

LERF provides:
- Scene images + camera poses (in COLMAP or transforms.json format)
- Text query annotations with ground-truth relevancy masks

Layout:
    dataset/lerf/{scene_name}/
        images/
        transforms.json  (NeRF-style, with camera params + file paths)
    dataset/lerf/{scene_name}/annotations/
        {query_text}/  (directory per query)
            frame_{idx}.png  (binary relevancy mask)
    output/radio_features/lerf/{scene_name}/
        backbone/rgb_{idx}.pt
"""

import json
import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from loc_gs.data.benchmark_paths import extract_feature_frame_index

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _load_mask(path: str) -> np.ndarray:
    """Load a binary mask as uint8 array (0/1)."""
    if _HAS_PIL:
        img = np.array(Image.open(path).convert("L"))
        return (img > 127).astype(np.uint8)
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (img > 127).astype(np.uint8)
    raise ImportError("Either PIL or cv2 is required for image loading")


def _resize_nearest(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    if _HAS_PIL:
        return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
    raise ImportError("Either PIL or cv2 is required for resizing")


# OpenGL (NeRF convention) → OpenCV camera convention
# OpenGL: +X right, +Y up, -Z forward
# OpenCV: +X right, +Y down, +Z forward
_GL_TO_CV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

_COLMAP_CAMERA_NUM_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


def _read_next_bytes(fid, num_bytes: int, format_sequence: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file")
    return struct.unpack("<" + format_sequence, data)


def _read_null_terminated_string(fid) -> str:
    chars = bytearray()
    while True:
        ch = fid.read(1)
        if ch == b"":
            raise EOFError("Unexpected end of COLMAP binary string")
        if ch == b"\x00":
            break
        chars.extend(ch)
    return chars.decode("utf-8")


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec.astype(np.float64)
    return np.array(
        [
            [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * w * z, 2.0 * x * z + 2.0 * w * y],
            [2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * w * x],
            [2.0 * x * z - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x * x - 2.0 * y * y],
        ],
        dtype=np.float32,
    )


def _camera_params_to_intrinsics(model_id: int, params: np.ndarray) -> tuple[float, float, float, float]:
    if model_id in {0, 2, 3, 7, 8, 9}:  # simple models use one focal length
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
        return fx, fy, cx, cy
    if model_id in {1, 4, 5, 6, 10}:  # pinhole / opencv variants
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
        return fx, fy, cx, cy
    raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")


def _read_cameras_binary(path: Path) -> Dict[int, Dict[str, object]]:
    cameras: Dict[int, Dict[str, object]] = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id = _read_next_bytes(fid, 8, "ii")
            width, height = _read_next_bytes(fid, 16, "QQ")
            num_params = _COLMAP_CAMERA_NUM_PARAMS.get(model_id)
            if num_params is None:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            params = np.array(
                _read_next_bytes(fid, 8 * num_params, f"{num_params}d"),
                dtype=np.float32,
            )
            cameras[int(camera_id)] = {
                "model_id": int(model_id),
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_images_binary(path: Path) -> List[Dict[str, object]]:
    images: List[Dict[str, object]] = []
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(fid, 4, "i")[0]
            qvec = np.array(_read_next_bytes(fid, 32, "4d"), dtype=np.float32)
            tvec = np.array(_read_next_bytes(fid, 24, "3d"), dtype=np.float32)
            camera_id = _read_next_bytes(fid, 4, "i")[0]
            name = _read_null_terminated_string(fid)
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * int(num_points2d), 1)
            images.append(
                {
                    "image_id": int(image_id),
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": int(camera_id),
                    "name": name,
                }
            )
    return images


def _resolve_scene_image_path(scene_root: Path, rel_path: str) -> str:
    rel = Path(rel_path)
    if rel.is_absolute():
        return str(rel)
    if (scene_root / rel).exists():
        return rel.as_posix()
    images_rel = Path("images") / rel.name
    if (scene_root / images_rel).exists():
        return images_rel.as_posix()
    return rel.as_posix()


def _parse_colmap_sparse(scene_root: Path) -> Dict:
    sparse_dir = scene_root / "sparse" / "0"
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    if not cameras_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"COLMAP sparse model not found under {sparse_dir} "
            f"(need cameras.bin and images.bin)"
        )

    cameras = _read_cameras_binary(cameras_path)
    images = _read_images_binary(images_path)
    if not images:
        raise ValueError(f"No images found in COLMAP sparse model: {images_path}")

    def _image_sort_key(entry: Dict[str, object]) -> tuple[int, int]:
        name = Path(str(entry["name"]))
        try:
            return (0, extract_feature_frame_index(name))
        except ValueError:
            return (1, int(entry["image_id"]))

    images.sort(key=_image_sort_key)

    c2w_list: List[np.ndarray] = []
    file_paths: List[str] = []
    for entry in images:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = _qvec_to_rotmat(np.asarray(entry["qvec"], dtype=np.float32))
        w2c[:3, 3] = np.asarray(entry["tvec"], dtype=np.float32)
        c2w_list.append(np.linalg.inv(w2c).astype(np.float32))
        file_paths.append(_resolve_scene_image_path(scene_root, str(entry["name"])))

    ref_camera = cameras[int(images[0]["camera_id"])]
    fx, fy, cx, cy = _camera_params_to_intrinsics(
        int(ref_camera["model_id"]),
        np.asarray(ref_camera["params"], dtype=np.float32),
    )
    return {
        "c2w_list": c2w_list,
        "file_paths": file_paths,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": int(ref_camera["width"]),
        "h": int(ref_camera["height"]),
    }


def _load_annotation_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_polygons(segmentation) -> List[np.ndarray]:
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return []

    first = segmentation[0]
    if isinstance(first, (int, float)):
        arr = np.asarray(segmentation, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 6 and arr.size % 2 == 0:
            return [arr.reshape(-1, 2)]
        return []

    if isinstance(first, (list, tuple)):
        arr = np.asarray(segmentation, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return [arr[:, :2]]

        polygons: List[np.ndarray] = []
        for poly in segmentation:
            poly_arr = np.asarray(poly, dtype=np.float32)
            if poly_arr.ndim == 1:
                if poly_arr.size < 6 or poly_arr.size % 2 != 0:
                    continue
                poly_arr = poly_arr.reshape(-1, 2)
            elif poly_arr.ndim == 2 and poly_arr.shape[1] >= 2:
                poly_arr = poly_arr[:, :2]
            else:
                continue
            if poly_arr.shape[0] >= 3:
                polygons.append(poly_arr)
        return polygons

    return []


def _rasterize_polygons(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if _HAS_CV2:
        for poly in polygons:
            pts = np.round(poly).astype(np.int32)
            if pts.shape[0] >= 3:
                cv2.fillPoly(mask, [pts], 1)
        return mask
    if _HAS_PIL:
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        for poly in polygons:
            pts = [tuple(map(float, p[:2])) for p in poly]
            if len(pts) >= 3:
                draw.polygon(pts, outline=1, fill=1)
        return (np.array(img) > 0).astype(np.uint8)
    raise ImportError("Either PIL or cv2 is required for polygon rasterization")


def _parse_transforms_json(path: str) -> Dict:
    """Parse a NeRF-style transforms.json file.

    Returns:
        dict with keys ``c2w_list`` (list of 4×4 np arrays in OpenCV convention),
        ``file_paths`` (list of image path strings), and camera parameters.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    c2w_list: List[np.ndarray] = []
    file_paths: List[str] = []

    for frame in frames:
        mat = np.array(frame["transform_matrix"], dtype=np.float32)
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0, 0, 0, 1]])
        # Convert OpenGL → OpenCV convention
        mat = mat @ _GL_TO_CV
        c2w_list.append(mat)
        file_paths.append(frame.get("file_path", ""))

    result = {
        "c2w_list": c2w_list,
        "file_paths": file_paths,
    }
    # Propagate camera params if present
    for key in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h"):
        if key in data:
            result[key] = data[key]

    return result


class LERFDataset(Dataset):
    """LERF dataset with pre-extracted RADIO features and text grounding annotations.

    Args:
        scene_root: Path to LERF scene (contains ``images/`` and ``transforms.json``).
        feature_dir: Path to pre-extracted RADIO features (contains ``backbone/``).
        annotation_dir: Optional path to text query annotations.
            If None, tries ``{scene_root}/annotations``.
        feature_height: Target spatial height for grounding masks.
        feature_width: Target spatial width for grounding masks.
    """

    def __init__(
        self,
        scene_root: str,
        feature_dir: str,
        annotation_dir: Optional[str] = None,
        feature_height: int = 30,
        feature_width: int = 40,
    ) -> None:
        super().__init__()
        self.scene_root = Path(scene_root)
        self.feature_dir = Path(feature_dir)
        self.feature_height = feature_height
        self.feature_width = feature_width

        # --- discover feature files ---
        backbone_dir = self.feature_dir / "backbone"
        if not backbone_dir.exists():
            backbone_dir = self.feature_dir
        self.feature_paths = sorted(
            backbone_dir.glob("rgb_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if len(self.feature_paths) == 0:
            raise FileNotFoundError(
                f"No rgb_*.pt features found in {backbone_dir}"
            )
        self.frame_indices = [
            int(p.stem.split("_")[1]) for p in self.feature_paths
        ]

        # --- load poses from transforms.json or raw COLMAP sparse model ---
        transforms_path = self.scene_root / "transforms.json"
        if transforms_path.exists():
            parsed = _parse_transforms_json(str(transforms_path))
            pose_source = transforms_path
        else:
            parsed = _parse_colmap_sparse(self.scene_root)
            pose_source = self.scene_root / "sparse" / "0"
        c2w_list = parsed["c2w_list"]
        self.file_paths = parsed["file_paths"]

        # Invert c2w → w2c, store as array
        self.poses_w2c = np.stack(
            [np.linalg.inv(c) for c in c2w_list], axis=0
        )
        logger.info("Loaded %d poses from %s", len(self.poses_w2c), pose_source)

        self.pose_by_frame_idx: Dict[int, np.ndarray] = {}
        for pose_w2c, file_path in zip(self.poses_w2c, self.file_paths):
            try:
                frame_idx = extract_feature_frame_index(Path(file_path))
            except ValueError:
                continue
            self.pose_by_frame_idx[int(frame_idx)] = pose_w2c

        # Store camera params for downstream use
        self.camera_params = {
            k: parsed[k]
            for k in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h")
            if k in parsed
        }

        # --- text query annotations ---
        if annotation_dir is not None:
            self.annotation_dir: Optional[Path] = Path(annotation_dir)
        elif (self.scene_root / "annotations").exists():
            self.annotation_dir = self.scene_root / "annotations"
        elif (self.scene_root.parent / "label" / self.scene_root.name).exists():
            self.annotation_dir = self.scene_root.parent / "label" / self.scene_root.name
        else:
            self.annotation_dir = None

        self.annotation_format: Optional[str] = None
        self.text_queries: List[str] = []
        if self.annotation_dir is not None and self.annotation_dir.exists():
            query_dirs = [d for d in self.annotation_dir.iterdir() if d.is_dir()]
            if query_dirs:
                self.annotation_format = "mask_dirs"
            elif any(self.annotation_dir.glob("frame_*.json")):
                self.annotation_format = "raw_polygons"
            self.text_queries = self.get_text_queries(str(self.annotation_dir))
            annotated_frame_ids = self.get_annotated_frame_ids(str(self.annotation_dir))
            if annotated_frame_ids:
                annotated_set = set(annotated_frame_ids)
                filtered = [
                    (path, frame_idx)
                    for path, frame_idx in zip(self.feature_paths, self.frame_indices)
                    if frame_idx in annotated_set
                ]
                if not filtered:
                    raise FileNotFoundError(
                        "No feature files matched the annotated LERF frames under "
                        f"{self.annotation_dir}"
                    )
                self.feature_paths = [path for path, _ in filtered]
                self.frame_indices = [frame_idx for _, frame_idx in filtered]

        logger.info(
            "LERFDataset: %d frames, %d text queries, annotations=%s, format=%s",
            len(self.feature_paths),
            len(self.text_queries),
            self.annotation_dir is not None,
            self.annotation_format,
        )

    def __len__(self) -> int:
        return len(self.feature_paths)

    @classmethod
    def get_text_queries(cls, annotation_dir: str) -> List[str]:
        """Discover text query strings from annotation subdirectories.

        Each subdirectory under ``annotation_dir`` is treated as a query.

        Args:
            annotation_dir: Path containing one subdirectory per query.

        Returns:
            Sorted list of query strings.
        """
        ann_path = Path(annotation_dir)
        if not ann_path.exists():
            return []
        queries = sorted(d.name for d in ann_path.iterdir() if d.is_dir())
        if queries:
            return queries

        json_files = sorted(
            ann_path.glob("frame_*.json"),
            key=extract_feature_frame_index,
        )
        if not json_files:
            return []

        categories = set()
        for path in json_files:
            data = _load_annotation_json(path)
            for obj in data.get("objects", []):
                category = str(obj.get("category", "")).strip()
                if category:
                    categories.add(category)
        return sorted(categories)

    @classmethod
    def get_annotated_frame_ids(cls, annotation_dir: str) -> List[int]:
        ann_path = Path(annotation_dir)
        if not ann_path.exists():
            return []

        json_files = sorted(
            ann_path.glob("frame_*.json"),
            key=extract_feature_frame_index,
        )
        if json_files:
            return [extract_feature_frame_index(path) for path in json_files]

        frame_ids = set()
        for query_dir in ann_path.iterdir():
            if not query_dir.is_dir():
                continue
            for mask_path in query_dir.glob("frame_*.png"):
                frame_ids.add(extract_feature_frame_index(mask_path))
        return sorted(frame_ids)

    @staticmethod
    def get_text_embeddings(
        query_texts: List[str],
        radio_model: Optional[object] = None,
    ) -> torch.Tensor:
        """Compute text embeddings for query strings.

        If ``radio_model`` is provided and has a ``encode_text`` method, uses it.
        Otherwise returns random unit vectors (for testing / stub).

        Args:
            query_texts: List of N query strings.
            radio_model: Optional model with ``encode_text(texts) → [N, D]``.

        Returns:
            Tensor of shape ``[N, D]`` (D=1280 by default).
        """
        n = len(query_texts)
        if n == 0:
            return torch.empty(0, 1280)

        if radio_model is not None and hasattr(radio_model, "encode_text"):
            embeddings = radio_model.encode_text(query_texts)
            if isinstance(embeddings, torch.Tensor):
                return embeddings.detach().cpu()
            return torch.from_numpy(np.array(embeddings))

        # Stub: deterministic pseudo-random embeddings for testing
        logger.warning(
            "radio_model not available; returning random text embeddings for %d queries",
            n,
        )
        gen = torch.Generator().manual_seed(hash(tuple(query_texts)) % (2**31))
        emb = torch.randn(n, 1280, generator=gen)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def _load_grounding_masks(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Load binary relevancy masks for all text queries at a given frame.

        Returns:
            [N_queries, H, W] float tensor, or None if annotations unavailable.
        """
        if self.annotation_dir is None or len(self.text_queries) == 0:
            return None

        if self.annotation_format == "raw_polygons":
            json_candidates = [
                self.annotation_dir / f"frame_{frame_idx:05d}.json",
                self.annotation_dir / f"frame_{frame_idx}.json",
            ]
            ann_path = next((path for path in json_candidates if path.exists()), None)
            if ann_path is None:
                return torch.zeros(
                    len(self.text_queries),
                    self.feature_height,
                    self.feature_width,
                    dtype=torch.float32,
                )

            data = _load_annotation_json(ann_path)
            info = data.get("info", {})
            mask_h = int(info.get("height", self.camera_params.get("h", self.feature_height)))
            mask_w = int(info.get("width", self.camera_params.get("w", self.feature_width)))
            query_masks = {
                query: np.zeros((mask_h, mask_w), dtype=np.uint8)
                for query in self.text_queries
            }
            for obj in data.get("objects", []):
                query = str(obj.get("category", "")).strip()
                if query not in query_masks:
                    continue
                polygons = _coerce_polygons(obj.get("segmentation"))
                if not polygons:
                    continue
                query_masks[query] = np.maximum(
                    query_masks[query],
                    _rasterize_polygons(polygons, mask_h, mask_w),
                )

            masks = []
            for query in self.text_queries:
                raw = _resize_nearest(query_masks[query], self.feature_height, self.feature_width)
                masks.append(torch.from_numpy(raw.astype(np.float32)))
            return torch.stack(masks, dim=0)

        masks = []
        for query in self.text_queries:
            mask_path = self.annotation_dir / query / f"frame_{frame_idx}.png"
            if mask_path.exists():
                raw = _load_mask(str(mask_path))
                raw = _resize_nearest(raw, self.feature_height, self.feature_width)
                masks.append(torch.from_numpy(raw.astype(np.float32)))
            else:
                masks.append(
                    torch.zeros(self.feature_height, self.feature_width, dtype=torch.float32)
                )

        return torch.stack(masks, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_idx = self.frame_indices[idx]

        # --- features ---
        radio_feat = torch.load(self.feature_paths[idx], map_location="cpu")
        if radio_feat.dim() == 4:
            radio_feat = radio_feat.squeeze(0)

        # --- pose ---
        pose_np = self.pose_by_frame_idx.get(frame_idx)
        if pose_np is None and 0 <= frame_idx < len(self.poses_w2c):
            pose_np = self.poses_w2c[frame_idx]
        if pose_np is None and 1 <= frame_idx <= len(self.poses_w2c):
            pose_np = self.poses_w2c[frame_idx - 1]
        if pose_np is None:
            logger.warning("Frame %d exceeds pose count; using identity", frame_idx)
            pose_w2c = torch.eye(4, dtype=torch.float32)
        else:
            pose_w2c = torch.from_numpy(np.array(pose_np, copy=True))

        out: Dict[str, torch.Tensor] = {
            "radio_features": radio_feat,
            "pose_w2c": pose_w2c,
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }

        # --- text queries + grounding masks ---
        if self.text_queries:
            out["text_queries"] = self.text_queries  # type: ignore[assignment]
            masks = self._load_grounding_masks(frame_idx)
            if masks is not None:
                out["grounding_masks"] = masks

        return out
