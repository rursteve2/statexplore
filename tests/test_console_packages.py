"""v1.1 Phases 7-8 (§14, §15, §20 N-O): R Console and local package manager."""
from __future__ import annotations

import io
import subprocess
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statexplore.core.console_import import apply_import, diff_import
from statexplore.core.importers import import_file
from statexplore.core.project import Project
from statexplore.rbridge import package_manager as P
from statexplore.rbridge.console import RConsole
from statexplore.rbridge.worker import REngineError

pytestmark = pytest.mark.r


@pytest.fixture
def user_lib_root(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEXPLORE_USER_LIB_ROOT", str(tmp_path / "userroot"))
    return tmp_path / "userroot"


@pytest.fixture
def small_ds(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50), "g": rng.choice([1.0, 2.0], 50)})
    p = tmp_path / "s.csv"
    df.to_csv(p, index=False)
    ds = import_file(p)
    ds.variables["g"].value_labels = {1.0: "low", 2.0: "high"}
    return ds


def kinds(res, kind):
    return [e.text for e in res.entries if e.kind == kind]


# ---- N. R Console ---------------------------------------------------------------------------------

def test_N_clean_vs_persistent_and_project_isolation(tmp_path, small_ds, user_lib_root):
    clean = RConsole(tmp_path / "c")
    clean.set_dataset(small_ds)
    r = clean.run("x <- 41 + 1\nx\ndim(my_data)\nlevels(my_data$g)")
    assert kinds(r, "output") == ["[1] 42", "[1] 50  2", '[1] "low"  "high"'] and not r.error
    assert kinds(clean.run("exists('x')"), "output") == ["[1] FALSE"]          # clean: nothing survives
    per = RConsole(tmp_path / "p", persistent=True)
    per.set_dataset(small_ds)
    per.run("x <- 42")
    assert kinds(per.run("exists('x')"), "output") == ["[1] TRUE"]
    assert any(o["name"] == "x" for o in per.objects()) and any(p["package"] == "base" for p in per.packages())
    per.clear_workspace()
    assert kinds(per.run("c(exists('x'), exists('my_data'))"), "output") == ["[1] FALSE  TRUE"]
    per.run("y <- 1")
    other = import_file(_csv(tmp_path, pd.DataFrame({"z": [1.0, 2.0, 3.0]})))
    per.set_dataset(other)                                                      # project switch ends the session
    assert kinds(per.run("c(exists('y'), nrow(my_data))"), "output") == ["[1] 0 3"]
    per.close()


def _csv(tmp_path, df, name="o.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_N_output_messages_warnings_errors_plots(tmp_path, small_ds, user_lib_root):
    c = RConsole(tmp_path / "c")
    c.set_dataset(small_ds)
    r = c.run("message('hello'); warning('careful')\nplot(my_data$x)\nlibrary(ggplot2)\n"
              "print(ggplot(my_data, aes(x)) + geom_histogram(bins = 10))\nprint('done')")
    assert kinds(r, "message") == ["hello"] and kinds(r, "warning") == ["Warning: careful"]
    assert len(r.plots) == 2 and all(p[:4] == b"\x89PNG" for p in r.plots)
    assert kinds(r, "output")[-1] == '[1] "done"' and kinds(r, "input")[0].startswith("message(")
    assert len(c.run("hist(my_data$x)").plots) == 1             # a single plot is still a list (JSON unboxing)
    r = c.run("a <- 1\nstop('boom')\nprint('never')")
    assert r.error and kinds(r, "error") == ["Error: boom"] and '[1] "never"' not in kinds(r, "output")
    r = c.run("f(")
    assert r.error and kinds(r, "error")[0].startswith("Syntax error")


def test_N_cancellation(tmp_path, small_ds, user_lib_root):
    c = RConsole(tmp_path / "c", persistent=True)
    c.set_dataset(small_ds)
    c.run("z <- 1")
    err: list[Exception] = []
    t0 = time.monotonic()
    th = threading.Thread(target=lambda: _catch(err, lambda: c.run("Sys.sleep(60)")))
    th.start()
    time.sleep(2.0)
    c.cancel()
    th.join(30)
    assert time.monotonic() - t0 < 20 and err and "stopped by the user" in str(err[0])
    assert kinds(c.run("exists('z')"), "output") == ["[1] FALSE"]               # session restarted
    c.close()


def _catch(store, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        store.append(e)


def test_N_internal_sessions_unaffected_and_explicit_import(tmp_path, small_ds, r_session, user_lib_root):
    from statexplore.analysis.builders import BY_KEY
    from statexplore.analysis.framework import run_analysis

    before, _ = run_analysis(r_session, small_ds, BY_KEY["t_one_sample"], {"dependent": ["x"]}, {})
    c = RConsole(tmp_path / "c", persistent=True)
    c.set_dataset(small_ds)
    c.run("dat <- NULL; my_data$x2 <- my_data$x * 2; my_data$g <- NULL; my_data <- my_data[1:40, ]")
    after, _ = run_analysis(r_session, small_ds, BY_KEY["t_one_sample"], {"dependent": ["x"]}, {})
    assert before.tables[0].rows == after.tables[0].rows                         # analysis worker untouched
    assert small_ds.n_rows == 50 and list(small_ds.variables) == ["x", "g"]         # dataset untouched
    csv, meta = c.export_my_data()
    diff, new_ds = diff_import(small_ds, csv, meta)
    assert (diff.orig_rows, diff.new_rows, diff.orig_cols, diff.new_cols) == (50, 40, 2, 2)
    assert diff.added == ["x2"] and diff.removed == ["g"]
    assert np.allclose(new_ds.raw["x2"], small_ds.raw["x"][:40] * 2, rtol=1e-12)
    proj = Project(small_ds)
    proj.add_result(before)
    proj.save(tmp_path / "orig.sxp")
    new_proj, snap = apply_import(proj, new_ds)
    assert snap.is_file() and Project.open(snap).dataset.n_rows == 50                # recoverable snapshot
    assert new_proj.dataset.n_rows == 40 and new_proj.results == proj.results        # history carried over
    assert proj.path == tmp_path / "orig.sxp"
    c.close()


def test_N_factor_columns_map_back_to_codes(tmp_path, small_ds, user_lib_root):
    c = RConsole(tmp_path / "c", persistent=True)
    c.set_dataset(small_ds)
    c.run("my_data$g2 <- my_data$g")
    csv, meta = c.export_my_data()
    diff, new_ds = diff_import(small_ds, csv, meta)
    assert new_ds.raw["g"].equals(small_ds.raw["g"]) and new_ds.var("g").value_labels == {1.0: "low", 2.0: "high"}
    assert "g2" in diff.added and new_ds.var("g2").storage.value == "string"
    c.close()


# ---- O. Package manager ---------------------------------------------------------------------------------

def _source_pkg(tmp_path: Path, name: str, imports: str = "", version: str = "0.1.0") -> Path:
    files = {
        f"{name}/DESCRIPTION": (f"Package: {name}\nVersion: {version}\nTitle: Test Package\nDescription: Test.\n"
                                "License: MIT\nAuthor: T\nMaintainer: T <t@example.org>\nNeedsCompilation: no\n"
                                + (f"Imports: {imports}\n" if imports else "")),
        f"{name}/NAMESPACE": "export(hello)\n" + (f"importFrom({imports.split()[0]}, describe)\n" if imports else ""),
        f"{name}/R/hello.R": f"hello <- function() \"hello from {name}\"\n",
    }
    tgz = tmp_path / f"{name}_{version}.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for n, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(n)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return tgz


def _binary_pkg(tmp_path: Path, name: str, imports: str = "", version: str = "0.1.0") -> Path:
    """Build a Windows binary .zip offline: install a pure-R source package into a scratch library, then zip
    the installed folder (the layout `R CMD INSTALL --build` produces; no zip tool is needed)."""
    from statexplore.rbridge.locate import find_rscript

    src = _source_pkg(tmp_path, name, version=version)
    lib = tmp_path / f"buildlib_{name}"
    lib.mkdir()
    code = f"install.packages('{src.as_posix()}', lib = '{lib.as_posix()}', repos = NULL, type = 'source')"
    subprocess.run([str(find_rscript()), "--vanilla", "-e", code], cwd=tmp_path, capture_output=True, timeout=300)
    assert (lib / name / "DESCRIPTION").is_file(), "test package build failed"
    z = tmp_path / f"{name}_{version}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as out:
        for f in (lib / name).rglob("*"):
            if f.is_file():
                out.write(f, f.relative_to(lib).as_posix())
    return _edit_description(z, {"Imports": imports}) if imports else z


def _edit_description(z: Path, fields: dict[str, str]) -> Path:
    out = z.with_name(z.stem + "_edited.zip")
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename.endswith("/DESCRIPTION") and item.filename.count("/") == 1:
                lines = [ln for ln in data.decode().splitlines() if ln.split(":")[0] not in fields]
                data = ("\n".join(lines + [f"{k}: {v}" for k, v in fields.items()]) + "\n").encode()
            dst.writestr(item, data)
    return out


def _rewrite_built(z: Path, new_built: str) -> Path:
    return _edit_description(z, {"Built": new_built})


def test_O_install_use_inventory_manifest_and_remove(tmp_path, user_lib_root):
    z = _binary_pkg(tmp_path, "sxhello")
    info = P.read_archive(z)
    assert info.kind == "binary" and info.package == "sxhello" and info.built_r.startswith("4.6")
    rep, log = P.install(z)
    lib = P.user_library()
    assert (lib / "sxhello").is_dir() and str(user_lib_root) in str(lib)
    inv = {r["package"]: r for r in P.inventory()}
    assert inv["sxhello"]["library"] == "user" and inv["psych"]["library"] == "core"
    m = P.manifest()
    assert m[0]["package"] == "sxhello" and m[0]["sha256"] == info.sha256 and m[0]["source_archive"] == z.name
    c = RConsole(tmp_path / "c")
    r = c.run("library(sxhello)\nhello()")
    assert kinds(r, "output") == ['[1] "hello from sxhello"'], r.entries
    P.remove("sxhello")
    assert not (lib / "sxhello").exists() and P.manifest() == []
    with pytest.raises(P.PackageError, match="core packages cannot be removed"):
        P.remove("psych")


def test_O_dependency_version_and_core_checks(tmp_path, user_lib_root):
    dep = _binary_pkg(tmp_path, "sxneedsdep", imports="sxnotinstalled")
    rep = P.check_archive(dep)
    assert not rep.ok and rep.missing == ["sxnotinstalled"] and "upload these package archives first" in rep.problems[0]
    with pytest.raises(P.PackageError, match="Missing dependencies"):
        P.install(dep)
    ok = _binary_pkg(tmp_path, "sxusespsych", imports="psych")             # satisfied by the core library
    assert P.check_archive(ok).ok
    old = _rewrite_built(_binary_pkg(tmp_path, "sxoldbuild"), "R 3.6.0; ; 2019-01-01 00:00:00 UTC; windows")
    assert any("built for R 3.6.0" in p for p in P.check_archive(old).problems)
    core = _binary_pkg(tmp_path, "psych", version="9.9.9")
    rep = P.check_archive(core)
    assert any("core R library" in p for p in rep.problems)
    with pytest.raises(P.PackageError):
        P.install(core)
    assert not (P.user_library() / "psych").exists()
    with pytest.raises(P.PackageError, match="Windows binary package"):
        P.read_archive(_csv(tmp_path, pd.DataFrame({"a": [1]}), "x.txt"))


def test_O_offline_by_construction():
    import inspect

    src = inspect.getsource(P.install)
    assert "repos = NULL" in src and "options(repos = NULL)" in src and "http" not in src
    assert "options(repos = NULL)" in (Path(P.__file__).parents[1] / "r" / "console_worker.R").read_text()
