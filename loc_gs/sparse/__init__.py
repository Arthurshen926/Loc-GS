"""Internal sparse-only localization primitives for the new Loc-GS mainline."""

from .audit import (
    ForbiddenRuntimeDependency,
    ForbiddenRuntimeHit,
    assert_internal_mainline_sources,
    reject_test_split,
    scan_forbidden_runtime_dependencies,
)
from .correspondences import SparseCandidateBatch
from .rerank import CandidateRerankConfig, rerank_candidate_rows, summarize_candidate_availability

__all__ = [
    "CandidateRerankConfig",
    "ForbiddenRuntimeDependency",
    "ForbiddenRuntimeHit",
    "SparseCandidateBatch",
    "assert_internal_mainline_sources",
    "reject_test_split",
    "rerank_candidate_rows",
    "scan_forbidden_runtime_dependencies",
    "summarize_candidate_availability",
]
