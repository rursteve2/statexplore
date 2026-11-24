"""Build an offline Windows distribution:  dist/StatExplore/
  StatExplore.exe (+ _internal/)   PyInstaller one-folder build
  R/                               copy of the R installation used in development
  rlib/                            StatExplore's R package library (not "r/library": Windows paths
                                   are case-insensitive and would merge it into R's own R/library)
  rlib/packages.lock.json          exact R and package versions
Then compile packaging/statexplore.iss with Inno Setup to produce the installer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Build output (~3 GB) can live on another drive: set STATEXPLORE_BUILD_DIR (contains dist/ and build/).
OUT = Path(os.environ.get("STATEXPLORE_BUILD_DIR", str(ROOT)))
DIST = OUT / "dist" / "StatExplore"


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    if DIST.exists():
        shutil.rmtree(DIST)
    from statexplore.rbridge.locate import DEFAULT_R_LIB, find_rscript

    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(OUT / "dist"),
                    "--workpath", str(OUT / "build"), str(ROOT / "packaging" / "statexplore.spec")], check=True)
    rscript = find_rscript()
    if rscript is None:
        sys.exit("R not found: install R or set STATEXPLORE_RSCRIPT")
    r_home = rscript.parents[2] if rscript.parent.name == "x64" else rscript.parents[1]
    print("Bundling R from", r_home)
    shutil.copytree(r_home, DIST / "R", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("doc", "tests", "*.html", "*.pdf"))
    shutil.copytree(DEFAULT_R_LIB, DIST / "rlib", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("doc", "tests", "help", "html"))
    lock = subprocess.run([str(rscript), "--vanilla", "-e",
                           f'ip <- installed.packages(lib.loc = c("{DEFAULT_R_LIB.as_posix()}", .Library));'
                           'cat(jsonlite::toJSON(list(R = R.version.string, packages = as.list(setNames(ip[, "Version"],'
                           ' ip[, "Package"]))), auto_unbox = TRUE))'],
                          capture_output=True, text=True, check=True,
                          env={**__import__("os").environ, "R_LIBS": str(DEFAULT_R_LIB)}).stdout
    (DIST / "rlib" / "packages.lock.json").write_text(json.dumps(json.loads(lock), indent=2))
    shutil.copy2(DIST / "rlib" / "packages.lock.json", ROOT / "r" / "packages.lock.json")
    for f in ("LICENSE", "THIRD-PARTY-NOTICES.md", "README.md"):
        shutil.copy2(ROOT / f, DIST / f)
    print("Done:", DIST)
    release_artifacts()


def release_artifacts() -> None:  # also: python packaging/build.py --release-only
    """Portable zip of dist/StatExplore and, if Inno Setup is installed, the installer."""
    import os

    sys.path.insert(0, str(ROOT / "src"))
    from statexplore import __version__

    out = OUT / "dist" / "release"
    out.mkdir(parents=True, exist_ok=True)
    zip_base = out / f"StatExplore-{__version__}-portable"
    print("Zipping", f"{zip_base}.zip")
    shutil.make_archive(str(zip_base), "zip", root_dir=OUT / "dist", base_dir="StatExplore")
    candidates = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
                  Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe"]
    iscc = next((c for c in candidates if c.is_file()), None)
    if iscc is None:
        print("Inno Setup not found: installer not built (portable zip only).")
        return
    subprocess.run([str(iscc), f"/O{out}", f"/DDistDir={OUT / 'dist'}", str(ROOT / "packaging" / "statexplore.iss")],
                   check=True)
    print("Release files:", *sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    if "--release-only" in sys.argv:
        for f in ("LICENSE", "THIRD-PARTY-NOTICES.md", "README.md"):
            shutil.copy2(ROOT / f, DIST / f)
        release_artifacts()
    else:
        main()
