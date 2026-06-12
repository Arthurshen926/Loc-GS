"""Internal sparse-only localization primitives for the new Loc-GS mainline."""

from .audit import (
    ForbiddenRuntimeDependency,
    ForbiddenRuntimeHit,
    assert_internal_mainline_sources,
    reject_test_split,
    scan_forbidden_runtime_dependencies,
)

__all__ = [
    "ForbiddenRuntimeDependency",
    "ForbiddenRuntimeHit",
    "assert_internal_mainline_sources",
    "reject_test_split",
    "scan_forbidden_runtime_dependencies",
]
