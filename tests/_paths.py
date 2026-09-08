"""Make ``import scission`` work from this checkout (flat package dir)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def package_root() -> Path:
    """This scission clone."""
    return Path(__file__).resolve().parents[1]


def ensure_scission() -> None:
    """Bind the local tree before ``pip install -e .``."""
    if "scission" in sys.modules:
        return
    pkg = package_root()
    init = pkg / "__init__.py"
    if not init.is_file():
        raise RuntimeError(f"scission package init not found at {init}")
    spec = importlib.util.spec_from_file_location(
        "scission",
        init,
        submodule_search_locations=[str(pkg)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scission from {pkg}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scission"] = mod
    spec.loader.exec_module(mod)
