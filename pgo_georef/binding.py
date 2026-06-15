"""Locate and import the CPU-built ``gtsam_points_py`` extension.

The extension links ``libgtsam_points.so`` (built alongside it) which is not on
the default loader path, so we preload it with ctypes before importing. GTSAM's
own shared libraries are expected on the system loader path (they are in the
``gtsam_docker`` image under ``/usr/local/lib``).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BUILD_DIRS = (
    _REPO_ROOT / "build_cpu",
    _REPO_ROOT / "build",
)

_module = None


def _find_extension(build_dir: Path) -> Optional[Path]:
    python_dir = build_dir / "python"
    if not python_dir.is_dir():
        return None
    matches = sorted(python_dir.glob("gtsam_points_py*.so"))
    return matches[0] if matches else None


def load_binding(build_dir: Optional[os.PathLike] = None):
    """Import and return the ``gtsam_points_py`` module (cached after first call)."""
    global _module
    if _module is not None:
        return _module

    search_dirs = [Path(build_dir)] if build_dir else list(_DEFAULT_BUILD_DIRS)
    env_dir = os.environ.get("GTSAM_POINTS_BUILD_DIR")
    if env_dir:
        search_dirs.insert(0, Path(env_dir))

    extension: Optional[Path] = None
    chosen_dir: Optional[Path] = None
    for candidate in search_dirs:
        candidate = candidate.expanduser().resolve()
        extension = _find_extension(candidate)
        if extension is not None:
            chosen_dir = candidate
            break

    if extension is None:
        searched = "\n  ".join(str(d) for d in search_dirs)
        raise ImportError(
            "Could not find gtsam_points_py*.so. Build the CPU binding first, e.g.\n"
            "  cmake -B build_cpu -DBUILD_WITH_CUDA=OFF -DBUILD_PYTHON=ON -DBUILD_DEMO=OFF\n"
            "  cmake --build build_cpu --target gtsam_points_py -j\n"
            f"Searched build dirs:\n  {searched}"
        )

    core_lib = chosen_dir / "libgtsam_points.so"
    if core_lib.is_file():
        ctypes.CDLL(str(core_lib), mode=ctypes.RTLD_GLOBAL)

    sys.path.insert(0, str(extension.parent))
    import gtsam_points_py  # noqa: E402

    _module = gtsam_points_py
    return _module
