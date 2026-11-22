"""Local R package manager (v1.1 §15): offline installation of user-supplied package archives.

- Packages go to a per-user, per-R-version library separate from StatExplore's core library;
  the core library always comes first on .libPaths(), so a user package can never mask a core one.
- Archives are untrusted: the DESCRIPTION is read and checked in Python before R ever sees the
  file (name, version, R version and platform it was built for, dependencies, core conflicts).
- Installation uses `install.packages(<file>, repos = NULL)`: nothing is downloaded.
- Every installation is recorded in a manifest next to the library (name, version, archive,
  SHA-256, date, library, dependencies).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .locate import find_rscript, r_library

BASE_PACKAGES = {"R", "base", "compiler", "datasets", "graphics", "grDevices", "grid", "methods", "parallel",
                 "splines", "stats", "stats4", "tcltk", "tools", "utils"}
MANIFEST = "statexplore-user-packages.json"
SECURITY_WARNING = ("Installing an R package runs code from the archive, now and whenever the package is loaded. "
                    "Install only packages from sources you trust. StatExplore installs from the local file only and "
                    "never downloads anything.")


class PackageError(Exception):
    """User-facing reason why a package cannot be installed or removed."""


@dataclass
class ArchiveInfo:
    path: str
    package: str
    version: str
    kind: str                         # binary | source
    built_r: str | None               # "4.6.2" for binaries
    built_platform: str | None        # "windows" ...
    needs_compilation: bool
    r_requirement: str | None
    depends: list[tuple[str, str | None]] = field(default_factory=list)   # (package, version constraint)
    sha256: str = ""


@dataclass
class CheckReport:
    info: ArchiveInfo
    problems: list[str] = field(default_factory=list)          # blocking
    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    existing_version: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def r_version_string(rscript: str | None = None) -> str:
    """'4.6.2' of the R installation StatExplore uses."""
    out = _rscript(["-e", "cat(paste(R.version$major, R.version$minor, sep = '.'))"], rscript, timeout=60)
    return out.stdout.strip().splitlines()[-1]


def user_library(r_version: str | None = None) -> Path:
    """%LOCALAPPDATA%\\StatExplore\\R\\<major.minor>\\user-library (packages are R-version specific)."""
    base = Path(os.environ.get("STATEXPLORE_USER_LIB_ROOT") or os.environ.get("LOCALAPPDATA") or Path.home())
    mm = ".".join((r_version or _cached_r_version()).split(".")[:2])
    return base / "StatExplore" / "R" / mm / "user-library"


_R_VERSION: str | None = None


def _cached_r_version() -> str:
    global _R_VERSION
    if _R_VERSION is None:
        try:
            _R_VERSION = r_version_string()
        except Exception:            # R missing: the console will report it when used
            _R_VERSION = "unknown"
    return _R_VERSION


# ---- archives ------------------------------------------------------------------------------------

def _parse_dcf(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    key = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line[0] in " \t" and key:
            out[key] += " " + line.strip()
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            out[key] = val.strip()
    return out


def _deps(fields: dict[str, str]) -> tuple[list[tuple[str, str | None]], str | None]:
    deps: list[tuple[str, str | None]] = []
    r_req = None
    for f in ("Depends", "Imports", "LinkingTo"):
        for part in filter(None, (p.strip() for p in fields.get(f, "").split(","))):
            m = re.match(r"^([A-Za-z0-9.]+)\s*(\((.*)\))?$", part)
            if not m:
                continue
            name, cons = m.group(1), (m.group(3) or None)
            if name == "R":
                r_req = cons
            elif name not in BASE_PACKAGES and name not in [d[0] for d in deps]:
                deps.append((name, cons))
    return deps, r_req


def read_archive(path: str | os.PathLike) -> ArchiveInfo:
    """Read DESCRIPTION from a Windows binary .zip or a source .tar.gz without extracting it."""
    p = Path(path)
    if not p.is_file():
        raise PackageError(f"File not found: {p}")
    name = p.name.lower()
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                desc = [n for n in z.namelist() if re.fullmatch(r"[^/]+/DESCRIPTION", n)]
                if len(desc) != 1:
                    raise PackageError("This .zip is not an R binary package (no single <package>/DESCRIPTION).")
                text = z.read(desc[0]).decode("utf-8", errors="replace")
            kind = "binary"
        elif name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(p, "r:gz") as t:
                desc = [m for m in t.getmembers() if re.fullmatch(r"[^/]+/DESCRIPTION", m.name)]
                if len(desc) != 1:
                    raise PackageError("This archive is not an R source package (no single <package>/DESCRIPTION).")
                text = t.extractfile(desc[0]).read().decode("utf-8", errors="replace")
            kind = "source"
        else:
            raise PackageError("Choose a Windows binary package (.zip) or a source package (.tar.gz).")
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
        raise PackageError(f"The archive cannot be read: {e}") from None
    f = _parse_dcf(text)
    if not f.get("Package") or not f.get("Version"):
        raise PackageError("DESCRIPTION has no Package/Version field.")
    built = f.get("Built", "")
    parts = [x.strip() for x in built.split(";")] if built else []
    built_r = parts[0].removeprefix("R ").strip() if parts else None
    platform = parts[3] if len(parts) > 3 else (parts[1] if len(parts) > 1 else None)
    if kind == "binary" and not built:
        raise PackageError("This .zip has no 'Built' field: it is not an installed (binary) R package.")
    deps, r_req = _deps(f)
    return ArchiveInfo(str(p), f["Package"], f["Version"], kind, built_r, platform,
                       f.get("NeedsCompilation", "no").lower() == "yes", r_req, deps,
                       hashlib.sha256(p.read_bytes()).hexdigest())


# ---- inventory ---------------------------------------------------------------------------------------

def inventory(rscript: str | None = None) -> list[dict[str, Any]]:
    """Installed packages in the core and user libraries (core first)."""
    lib = user_library()
    code = (
        "libs <- c(core = Sys.getenv('STATEXPLORE_R_LIB'), user = Sys.getenv('STATEXPLORE_USER_LIB'))\n"
        "libs <- libs[nzchar(libs) & dir.exists(libs)]\n"
        "ip <- utils::installed.packages(lib.loc = c(libs, .Library), fields = 'Built')\n"
        "np <- function(x) tolower(normalizePath(x, '/', FALSE))\n"
        "lp <- np(ip[, 'LibPath'])\n"
        "where <- ifelse(lp == np(libs['core']), 'core', ifelse('user' %in% names(libs) & lp == np(libs['user']), 'user', 'R'))\n"
        "out <- lapply(seq_len(nrow(ip)), function(i) list(package = ip[i, 'Package'], version = ip[i, 'Version'],"
        " library = where[i], path = ip[i, 'LibPath'], depends = paste(na.omit(ip[i, c('Depends', 'Imports', 'LinkingTo')]),"
        " collapse = ', '), built = ip[i, 'Built']))\n"
        "cat(jsonlite::toJSON(out, auto_unbox = TRUE, null = 'null', na = 'null'))\n"
    )
    out = _rscript(["-e", code], rscript, extra_env={"STATEXPLORE_USER_LIB": str(lib)}, timeout=120)
    text = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "[]"
    rows = json.loads(text)
    seen: set[str] = set()
    res = []
    for r in rows:                      # .libPaths order: the first copy is the one R loads
        r["active"] = r["package"] not in seen
        seen.add(r["package"])
        res.append(r)
    return res


def manifest(lib: Path | None = None) -> list[dict[str, Any]]:
    p = (lib or user_library()) / MANIFEST
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _write_manifest(lib: Path, entries: list[dict[str, Any]]) -> None:
    (lib / MANIFEST).write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ---- checks ----------------------------------------------------------------------------------------

def _vkey(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.split(r"[.-]", v) if x.isdigit())


def _satisfies(version: str, constraint: str | None) -> bool:
    if not constraint:
        return True
    m = re.match(r"^\s*(>=|<=|==|>|<)\s*([0-9.\-]+)\s*$", constraint)
    if not m:
        return True
    op, want = m.groups()
    a, b = _vkey(version), _vkey(want)
    return {">=": a >= b, "<=": a <= b, "==": a == b, ">": a > b, "<": a < b}[op]


def check_archive(path: str | os.PathLike, rscript: str | None = None,
                  installed: list[dict[str, Any]] | None = None) -> CheckReport:
    """Everything that can be checked before installation (§15.3)."""
    info = read_archive(path)
    rep = CheckReport(info)
    rv = r_version_string(rscript)
    installed = installed if installed is not None else inventory(rscript)
    active = {r["package"]: r for r in installed if r["active"]}
    core = {r["package"] for r in installed if r["library"] in ("core", "R")}
    if info.package in core:
        rep.problems.append(f"'{info.package}' is part of StatExplore's core R library (or R itself). Installing another "
                            "version could change StatExplore's analyses, so it is not allowed.")
    if info.kind == "binary":
        if info.built_platform and "windows" not in info.built_platform.lower() and "mingw" not in info.built_platform.lower():
            rep.problems.append(f"This binary was built for '{info.built_platform}', not Windows.")
        if info.built_r and info.built_r.split(".")[:2] != rv.split(".")[:2]:
            rep.problems.append(f"This binary was built for R {info.built_r}; StatExplore uses R {rv}. Binary packages "
                                "only work with the R version (major.minor) they were built for.")
    else:
        rep.notes.append("Source package: it will be built from source.")
        if info.needs_compilation and not _has_build_tools(rscript):
            rep.problems.append("This source package contains compiled code and needs Rtools (a C/C++ compiler), which "
                                "was not found. Use the Windows binary (.zip) of the package instead.")
    if info.r_requirement and not _satisfies(rv, info.r_requirement):
        rep.problems.append(f"The package requires R {info.r_requirement}; this is R {rv}.")
    for dep, cons in info.depends:
        have = active.get(dep)
        if have is None:
            rep.missing.append(dep + (f" ({cons})" if cons else ""))
        elif not _satisfies(have["version"], cons):
            rep.missing.append(f"{dep} ({cons}; installed {have['version']})")
    if rep.missing:
        rep.problems.append("Missing dependencies — upload these package archives first: " + ", ".join(rep.missing))
    if info.package in active and info.package not in core:
        rep.existing_version = active[info.package]["version"]
        rep.notes.append(f"Version {rep.existing_version} is installed in your library and will be replaced.")
    return rep


def _has_build_tools(rscript: str | None) -> bool:
    try:
        out = _rscript(["-e", "cat(isTRUE(pkgbuild::has_build_tools(debug = FALSE)))"], rscript, timeout=120)
        return out.stdout.strip().endswith("TRUE")
    except Exception:
        return False


# ---- install / remove ---------------------------------------------------------------------------------

def install(path: str | os.PathLike, rscript: str | None = None) -> tuple[CheckReport, str]:
    """Install a checked local archive into the user library; returns (report, installation log).
    Raises PackageError with the reasons if any check fails."""
    rep = check_archive(path, rscript)
    if not rep.ok:
        raise PackageError("\n".join(rep.problems))
    lib = user_library(r_version_string(rscript))
    lib.mkdir(parents=True, exist_ok=True)
    kind = "win.binary" if rep.info.kind == "binary" else "source"
    code = (
        "options(repos = NULL)\n"
        f"utils::install.packages({json.dumps(Path(path).as_posix())}, lib = {json.dumps(lib.as_posix())}, "
        f"repos = NULL, type = {json.dumps(kind)})\n"
        f".libPaths(c(Sys.getenv('STATEXPLORE_R_LIB'), {json.dumps(lib.as_posix())}, .libPaths()))\n"
        f"ok <- requireNamespace({json.dumps(rep.info.package)}, quietly = TRUE)\n"
        "cat('\\n@@SX_INSTALL_OK@@', ok, '\\n')\n"
    )
    out = _rscript(["-e", code], rscript, timeout=900)
    log_text = (out.stdout + "\n" + out.stderr).strip()
    if "@@SX_INSTALL_OK@@ TRUE" not in out.stdout:
        raise PackageError("Installation failed. R reported:\n" + log_text[-4000:])
    entries = [e for e in manifest(lib) if e["package"] != rep.info.package]
    entries.append({"package": rep.info.package, "version": rep.info.version, "kind": rep.info.kind,
                    "source_archive": Path(path).name, "sha256": rep.info.sha256,
                    "installed_at": _dt.datetime.now().isoformat(timespec="seconds"), "library": str(lib),
                    "depends": [d for d, _ in rep.info.depends], "built": rep.info.built_r})
    _write_manifest(lib, entries)
    return rep, log_text


def remove(package: str, rscript: str | None = None) -> str:
    """Remove a package from the USER library only, unless another user package depends on it."""
    lib = user_library(r_version_string(rscript))
    inv = inventory(rscript)
    row = next((r for r in inv if r["package"] == package and r["library"] == "user"), None)
    if row is None:
        raise PackageError(f"'{package}' is not in your user library (core packages cannot be removed).")
    users = [r["package"] for r in inv if r["library"] == "user" and r["package"] != package
             and re.search(rf"(^|[\s,(]){re.escape(package)}([\s,(]|$)", r.get("depends") or "")]
    if users:
        raise PackageError(f"'{package}' is needed by: {', '.join(users)}. Remove those first.")
    out = _rscript(["-e", f"utils::remove.packages({json.dumps(package)}, lib = {json.dumps(lib.as_posix())})"],
                   rscript, timeout=120)
    _write_manifest(lib, [e for e in manifest(lib) if e["package"] != package])
    return (out.stdout + out.stderr).strip()


def _rscript(args: list[str], rscript: str | None, extra_env: dict[str, str] | None = None,
             timeout: float = 300) -> subprocess.CompletedProcess:
    """Run Rscript with an argument list (no shell) and StatExplore's libraries on the path."""
    exe = find_rscript(rscript)
    if exe is None:
        raise PackageError("R was not found.")
    env = os.environ.copy()
    env["STATEXPLORE_R_LIB"] = str(r_library())
    env.setdefault("STATEXPLORE_USER_LIB", str(user_library()) if _R_VERSION else "")
    env["R_LIBS_USER"] = os.devnull if sys.platform != "win32" else "NUL"     # ignore the personal R library
    env.update(extra_env or {})
    pre = ".libPaths(c(Sys.getenv('STATEXPLORE_R_LIB'), .libPaths()))\n"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with tempfile.TemporaryDirectory() as tmp:
        if args and args[0] == "-e":     # code goes in a file: no multi-line command-line arguments, no shell
            script = Path(tmp) / "sx_pkg.R"
            script.write_text(pre + args[1], encoding="utf-8")
            args = [str(script)]
        return subprocess.run([str(exe), "--vanilla", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, env=env, cwd=tmp, creationflags=flags)
