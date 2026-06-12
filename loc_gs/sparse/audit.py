from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORBIDDEN_RUNTIME_PATTERNS: tuple[str, ...] = (
    "/root/ULF-Loc",
    "from ulfloc import",
    "import ulfloc",
    "third_party/stdloc/stdloc.py",
    "third_party/stdloc/train.py",
)


class ForbiddenRuntimeDependency(RuntimeError):
    """Raised when new-mainline code depends on external method runtime code."""


@dataclass(frozen=True)
class ForbiddenRuntimeHit:
    path: Path
    pattern: str
    line_number: int
    line: str


def reject_test_split(split_name: str | None, *, purpose: str) -> str:
    split = str(split_name or "unknown").strip() or "unknown"
    lowered = split.lower()
    if lowered == "test" or lowered == "official_test" or lowered.endswith("_test"):
        raise ValueError(f"test split is not allowed for {purpose}: {split}")
    return split


def scan_forbidden_runtime_dependencies(paths: Iterable[str | Path]) -> list[ForbiddenRuntimeHit]:
    hits: list[ForbiddenRuntimeHit] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            for pattern in FORBIDDEN_RUNTIME_PATTERNS:
                if pattern in line:
                    hits.append(
                        ForbiddenRuntimeHit(
                            path=path,
                            pattern=pattern,
                            line_number=line_number,
                            line=line.strip(),
                        )
                    )
    return hits


def assert_internal_mainline_sources(paths: Iterable[str | Path]) -> None:
    hits = scan_forbidden_runtime_dependencies(paths)
    if not hits:
        return
    detail = "; ".join(f"{hit.path}:{hit.line_number}: {hit.pattern}" for hit in hits)
    raise ForbiddenRuntimeDependency(f"forbidden external runtime dependency found: {detail}")
