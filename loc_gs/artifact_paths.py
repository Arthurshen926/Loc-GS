from __future__ import annotations

from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SIGLIP2_TEXT_EMBEDDINGS = "checkpoints/siglip2_text_embeddings_v2.pt"
DEFAULT_SIGLIP2_PROJECTION_WEIGHTS = "checkpoints/siglip2_feat_projection.pth"

LEGACY_SIGLIP2_TEXT_EMBEDDINGS = "output/loc_gs/siglip2_text_embeddings_v2.pt"
LEGACY_SIGLIP2_PROJECTION_WEIGHTS = "output/loc_gs/siglip2_feat_projection.pth"


def resolve_repo_artifact_path(
    path: str | Path,
    fallback_rel_paths: Iterable[str] = (),
) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []
    if str(path):
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(REPO_ROOT / raw_path)
            candidates.append(raw_path)

    for rel_path in fallback_rel_paths:
        candidates.append(REPO_ROOT / rel_path)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    if candidates:
        return candidates[0]
    raise ValueError("No artifact path candidates provided")


def resolve_siglip_text_embeddings_path(path: str | Path) -> Path:
    return resolve_repo_artifact_path(
        path,
        fallback_rel_paths=(
            DEFAULT_SIGLIP2_TEXT_EMBEDDINGS,
            LEGACY_SIGLIP2_TEXT_EMBEDDINGS,
        ),
    )


def resolve_siglip_projection_path(path: str | Path) -> Path:
    return resolve_repo_artifact_path(
        path,
        fallback_rel_paths=(
            DEFAULT_SIGLIP2_PROJECTION_WEIGHTS,
            LEGACY_SIGLIP2_PROJECTION_WEIGHTS,
        ),
    )
