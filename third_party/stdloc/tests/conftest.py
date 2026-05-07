from __future__ import annotations

import sys
from pathlib import Path


STDLOC_ROOT = Path(__file__).resolve().parents[1]
if str(STDLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(STDLOC_ROOT))


def pytest_configure() -> None:
    import os

    os.chdir(STDLOC_ROOT)
