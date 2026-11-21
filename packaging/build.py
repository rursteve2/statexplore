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
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "StatExplore"


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    if DIST.exists():
        shutil.rmtree(DIST)
    from statexplore.rbridge.locate import DEFAULT_R_LIB, find_rscript

    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(ROOT / "dist"),
                    "--workpath", str(ROOT / "build"), str(ROOT / "packaging" / "statexplore.spec")], check=True)
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
    print("Done:", DIST)


if __name__ == "__main__":
    main()
