from __future__ import annotations

SPARSE_MATCHERS = ("topk", "stdloc_parity", "lightglue", "dim")
DENSE_MATCHERS = ("topk", "stdloc_parity", "lightglue_rendered")
DIM_PIPELINES = (
    "superpoint+lightglue",
    "aliked+lightglue",
    "disk+lightglue",
    "sift+kornia_matcher",
    "keynetaffnethardnet+kornia_matcher",
    "loftr",
    "roma",
)


def normalize_dim_pipeline(pipeline: str) -> str:
    return str(pipeline).strip().lower().replace(" ", "")


def resolve_sparse_dense_matchers(
    matcher: str,
    sparse_matcher: str = "",
    dense_matcher: str = "",
) -> tuple[str, str]:
    legacy = str(matcher or "topk")
    sparse = str(sparse_matcher or legacy)
    dense = str(dense_matcher or legacy)
    return sparse, dense
