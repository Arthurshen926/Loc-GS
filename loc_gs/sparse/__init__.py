"""Internal sparse-only localization primitives for the new Loc-GS mainline."""

from .audit import (
    ForbiddenRuntimeDependency,
    ForbiddenRuntimeHit,
    assert_internal_mainline_sources,
    reject_test_split,
    scan_forbidden_runtime_dependencies,
)
from .correspondences import SparseCandidateBatch
from .failure_profile import SparseFailureProfileConfig, build_sparse_failure_profile
from .pipeline import (
    SparseLocalizationConfig,
    SparseLocalizationInput,
    SparseLocalizationResult,
    SparsePipelineCandidate,
    run_sparse_localization,
)
from .rerank import CandidateRerankConfig, rerank_candidate_rows, summarize_candidate_availability

__all__ = [
    "CandidateRerankConfig",
    "ForbiddenRuntimeDependency",
    "ForbiddenRuntimeHit",
    "SparseCandidateBatch",
    "SparseFailureProfileConfig",
    "SparseLocalizationConfig",
    "SparseLocalizationInput",
    "SparseLocalizationResult",
    "SparsePipelineCandidate",
    "assert_internal_mainline_sources",
    "build_sparse_failure_profile",
    "reject_test_split",
    "rerank_candidate_rows",
    "run_sparse_localization",
    "scan_forbidden_runtime_dependencies",
    "summarize_candidate_availability",
]
