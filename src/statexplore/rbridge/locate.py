"""Locate an R installation (Rscript.exe) and the project-local R library."""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
# In an installed (PyInstaller) build, R and its library are bundled next to the executable.
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[3]
PROJECT_ROOT = APP_DIR
# Windows paths are case-insensitive: the bundled library must not be "r/library", which is R's own "R/library".
DEFAULT_R_LIB = APP_DIR / ("rlib" if FROZEN else "r/library")
BUNDLED_RSCRIPT = APP_DIR / "R" / "bin" / "x64" / "Rscript.exe"
WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "r" / "worker.R"

PHASE1_PACKAGES = ["jsonlite", "psych", "ggplot2"]
PHASE2_PACKAGES = ["car", "afex", "emmeans", "effectsize", "ppcor", "lmtest", "detectseparation"]
PHASE5_PACKAGES = ["lme4", "lmerTest", "nlme", "mmrm", "performance"]
REQUIRED_PACKAGES = PHASE1_PACKAGES + PHASE2_PACKAGES + PHASE5_PACKAGES


def _version_key(path: Path) -> tuple[int, ...]:
    m = re.search(r"R-(\d+)\.(\d+)\.(\d+)", str(path))
    return tuple(int(x) for x in m.groups()) if m else (0,)


def _registry_candidates() -> list[Path]:
    if sys.platform != "win32":
        return []
    import winreg

    found: list[Path] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for key in (r"SOFTWARE\R-core\R64", r"SOFTWARE\R-core\R"):
            try:
                with winreg.OpenKey(hive, key) as k:
                    install, _ = winreg.QueryValueEx(k, "InstallPath")
                    found.append(Path(install) / "bin" / "Rscript.exe")
            except OSError:
                continue
    return found


def _directory_candidates() -> list[Path]:
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "R",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "R",
        Path.home() / "AppData" / "Local" / "Programs" / "R",
    ]
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(root.glob("R-*/bin/Rscript.exe"))  # mapped to bin/x64 by _real_rscript
    return sorted(found, key=_version_key, reverse=True)


def find_rscript(configured: str | None = None) -> Path | None:
    """Resolution order: explicit setting, STATEXPLORE_RSCRIPT, PATH, registry, standard dirs."""
    candidates: list[Path] = [BUNDLED_RSCRIPT] if FROZEN else []
    for explicit in (configured, os.environ.get("STATEXPLORE_RSCRIPT")):
        if explicit:
            candidates.append(Path(explicit))
    on_path = shutil.which("Rscript")
    if on_path:
        candidates.append(Path(on_path))
    candidates += _registry_candidates() + _directory_candidates()
    for c in candidates:
        if c.is_file():
            return _real_rscript(c)
    return None


def _real_rscript(path: Path) -> Path:
    """On Windows, R/bin/Rscript.exe is a launcher that spawns bin/x64/Rscript.exe via cmd.exe.
    Use the real binary so the process we control (and may kill) is R itself."""
    x64 = path.parent / "x64" / path.name
    return x64 if path.parent.name.lower() == "bin" and x64.is_file() else path


def r_library() -> Path:
    return Path(os.environ.get("STATEXPLORE_R_LIB", DEFAULT_R_LIB))
