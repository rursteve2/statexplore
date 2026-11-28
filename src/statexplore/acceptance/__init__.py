"""Consolidated acceptance runner (v1.3.6): the automatable parts of owner tests 55-61 in one unattended run.

    StatExplore.exe --acceptance [--fixtures DIR] [--out DIR] [--only 55,57] [--quick]
    python -m statexplore --acceptance ...          (source checkout)

Every check records the case, what was checked, the expected and actual values, the tolerance, the execution kind and
PASS / FAIL / BLOCKED. Execution kinds are labelled, never mixed up:
  app+R     the application ran real R fits (same code paths as the GUI, no window)
  oracle-R  an independent R calculation written here (base R / geepack / lme4 calls, not the app's code)
  app       application logic without R (identity, grammar, exports, ledgers)
  script-R  a script EXPORTED by the application, run in a fresh Rscript (exit status and its own check report)
A FAIL is a FAIL; a check that could not run is BLOCKED with the reason (never PASS). The run writes one timestamped
evidence ZIP (results.csv / results.json / summary.md / environment.json / logs / artifacts / SHA256SUMS.txt). It uses
copies of the fixtures in a private work folder; nothing is installed, uploaded or changed elsewhere.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable

KINDS = ("app+R", "oracle-R", "app", "script-R")


class Ctx:
    def __init__(self, fixtures: Path, work: Path, quick: bool):
        self.fixtures, self.work, self.quick = fixtures, work, quick
        self.rows: list[dict[str, Any]] = []
        self.case = ""
        self.log_lines: list[str] = []
        self._rscript: Path | None = None
        self.live: Path | None = None

    # ---- recording ----
    def log(self, text: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {self.case}: {text}"
        self.log_lines.append(line)
        print(line, flush=True)
        if self.live is not None:        # progress file (the windowed executable has no console)
            with open(self.live, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def check(self, name: str, expected: Any, actual: Any, *, ok: bool | None = None, tol: str = "exact",
              kind: str = "app", evidence: str = "") -> bool:
        if ok is None:
            ok = expected == actual
        self.rows.append({"case": self.case, "check": name, "expected": _txt(expected), "actual": _txt(actual),
                          "tolerance": tol, "kind": kind, "status": "PASS" if ok else "FAIL", "evidence": evidence})
        self.log(f"{'PASS' if ok else 'FAIL'} {name}: expected {_txt(expected)}, actual {_txt(actual)}")
        return bool(ok)

    def close(self, name: str, expected: float, actual: Any, abs_tol: float, kind: str = "app+R") -> bool:
        ok = actual is not None and isinstance(actual, (int, float)) and math.isfinite(actual) and \
            abs(float(actual) - expected) <= abs_tol
        return self.check(name, expected, actual, ok=ok, tol=f"abs {abs_tol:g}", kind=kind)

    def blocked(self, name: str, why: str) -> None:
        self.rows.append({"case": self.case, "check": name, "expected": "", "actual": why, "tolerance": "",
                          "kind": "", "status": "BLOCKED", "evidence": ""})
        self.log(f"BLOCKED {name}: {why}")

    # ---- resources ----
    def fx(self, name: str) -> Path:
        p = self.fixtures / name
        if not p.is_file():
            raise FileNotFoundError(f"fixture {name} not found in {self.fixtures}")
        dst = self.work / "inputs" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(p, dst)            # always a copy: the original fixture is never touched
        return dst

    def dir(self, name: str) -> Path:
        d = self.work / "artifacts" / self.case / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rscript(self) -> Path:
        if self._rscript is None:
            from ..rbridge.locate import find_rscript

            self._rscript = find_rscript()
            if self._rscript is None:
                raise RuntimeError("R not found (bundled R, STATEXPLORE_RSCRIPT, PATH or registry)")
        return self._rscript

    def r_env(self) -> dict[str, str]:
        from ..rbridge.locate import r_library

        lib = str(r_library())
        return dict(os.environ, R_LIBS=lib, R_LIBS_USER=lib, OPENBLAS_NUM_THREADS="1")

    def run_r(self, cwd: Path, script: str, timeout: int = 3600) -> subprocess.CompletedProcess:
        r = subprocess.run([str(self.rscript), "--vanilla", script], cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=self.r_env(),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        logs = self.work / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{self.case}_{Path(script).stem}_{int(time.time() * 1000) % 10**8}.log").write_text(
            f"cwd: {cwd}\nexit: {r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}", encoding="utf-8")
        return r

    def oracle(self, code: str, name: str = "oracle") -> subprocess.CompletedProcess:
        d = self.dir("oracle")
        (d / f"{name}.R").write_text(code, encoding="utf-8")
        return self.run_r(d, f"{name}.R")

    def worker(self):
        from ..rbridge.worker import RWorker

        w = RWorker(str(self.rscript))
        w.start()
        return w


def _txt(x: Any) -> str:
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, (dict, list, tuple)):
        try:
            return json.dumps(x, default=str)[:400]
        except Exception:  # noqa: BLE001
            return str(x)[:400]
    return str(x)[:400]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- environment ------------------------------------------------------------------------------------------------------

def environment(ctx: Ctx) -> dict[str, Any]:
    from .. import __version__
    from ..export.exporters import build_id

    env: dict[str, Any] = {"statexplore": __version__, "build": build_id(), "frozen": bool(getattr(sys, "frozen", False)),
                           "executable": sys.executable, "python": platform.python_version(),
                           "os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                           "started": _dt.datetime.now().isoformat(timespec="seconds"), "quick": ctx.quick,
                           "fixtures": str(ctx.fixtures)}
    try:
        env["executable_sha256"] = _sha(Path(sys.executable))
    except OSError:
        pass
    try:
        import numpy as np
        import pandas as pd

        env["numpy"], env["pandas"] = np.__version__, pd.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..discovery.engine import ENGINE_PACKAGES

        w = ctx.worker()
        try:
            env["r"] = w.request("versions", {"packages": ENGINE_PACKAGES + ["geepack", "lme4"]}).result
        finally:
            w.stop()
        env["rscript"] = str(ctx.rscript)
    except Exception as e:  # noqa: BLE001
        env["r"] = f"unavailable: {e}"
    env["fixture_sha256"] = {p.name: _sha(p) for p in sorted(ctx.fixtures.glob("*")) if p.is_file()}
    return env


# ---- driver -----------------------------------------------------------------------------------------------------------

def run(fixtures: Path, out: Path, only: set[str] | None = None, quick: bool = False) -> Path:
    from . import cases

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(tempfile.mkdtemp(prefix=f"sx_acceptance_{stamp}_"))
    ctx = Ctx(Path(fixtures), work, quick)
    out.mkdir(parents=True, exist_ok=True)
    ctx.live = out / f"acceptance_live_{stamp}.log"
    ctx.case = "env"
    env = environment(ctx)
    (work / "environment.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")
    timings = {}
    for cid, fn in cases.CASES:
        if only and cid not in only:
            continue
        ctx.case = cid
        t0 = time.perf_counter()
        ctx.log(f"start ({fn.__doc__.splitlines()[0] if fn.__doc__ else ''})")
        try:
            fn(ctx)
        except Exception as e:  # noqa: BLE001 - a crash is a FAIL of that case, the run continues
            ctx.rows.append({"case": cid, "check": "case ran to completion", "expected": "no error",
                             "actual": f"{type(e).__name__}: {e}", "tolerance": "", "kind": "", "status": "FAIL",
                             "evidence": ""})
            ctx.log("FAIL (crash)\n" + traceback.format_exc())
        timings[cid] = round(time.perf_counter() - t0, 1)
    return _package(ctx, out, stamp, env, timings)


def _package(ctx: Ctx, out: Path, stamp: str, env: dict[str, Any], timings: dict[str, float]) -> Path:
    import csv

    work = ctx.work
    cols = ["case", "check", "expected", "actual", "tolerance", "kind", "status", "evidence"]
    with open(work / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(ctx.rows)
    (work / "results.json").write_text(json.dumps({"environment": env, "timings_s": timings, "results": ctx.rows},
                                                  indent=1, default=str), encoding="utf-8")
    (work / "run.log").write_text("\n".join(ctx.log_lines), encoding="utf-8")
    by = {}
    for r in ctx.rows:
        by.setdefault(r["case"], {"PASS": 0, "FAIL": 0, "BLOCKED": 0})[r["status"]] += 1
    tot = {k: sum(v[k] for v in by.values()) for k in ("PASS", "FAIL", "BLOCKED")}
    lines = [f"# StatExplore {env.get('statexplore')} acceptance run {stamp}", "",
             f"Build: {env.get('build')} - frozen: {env.get('frozen')} - R: "
             f"{(env.get('r') or {}).get('r_version') if isinstance(env.get('r'), dict) else env.get('r')}", "",
             f"**Overall: {tot['PASS']} PASS, {tot['FAIL']} FAIL, {tot['BLOCKED']} BLOCKED**", "",
             "| Case | PASS | FAIL | BLOCKED | seconds |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {c} | {v['PASS']} | {v['FAIL']} | {v['BLOCKED']} | {timings.get(c, '')} |" for c, v in by.items()]
    bad = [r for r in ctx.rows if r["status"] != "PASS"]
    if bad:
        lines += ["", "## Not passed", ""] + [f"- {r['case']} {r['status']}: {r['check']} - expected {r['expected']}, "
                                               f"actual {r['actual']}" for r in bad]
    lines += ["", "Kinds: app+R = application with real R; oracle-R = independent R calculation; app = application "
              "logic without R; script-R = exported script in a fresh Rscript.", "",
              "Remaining manual checks: see manual_acceptance_v1.3.6.md (installed GUI only)."]
    (work / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sums = [f"{_sha(p)}  {p.relative_to(work).as_posix()}" for p in sorted(work.rglob("*"))
            if p.is_file() and "inputs" not in p.relative_to(work).parts]
    (work / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    z = out / f"StatExplore_v{env.get('statexplore')}_Acceptance_Results_{stamp}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(work.rglob("*")):
            rel = p.relative_to(work)
            if p.is_file() and "inputs" not in rel.parts and p.stat().st_size < 60_000_000:
                zf.write(p, rel.as_posix())
    shutil.copy(work / "summary.md", out / f"Acceptance_Summary_{stamp}.md")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nEvidence: {z}\n" + "\n".join(lines[:12]), flush=True)
    return z


def main(argv: list[str]) -> int:
    def opt(name: str, default: str | None = None) -> str | None:
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default
    here = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
    fixtures = Path(opt("--fixtures") or next((p for p in (here / "fixtures", Path.cwd() / "fixtures") if p.is_dir()),
                                              here / "fixtures"))
    out = Path(opt("--out") or (Path.home() / "Documents" / "StatExplore_Acceptance"))
    only = set((opt("--only") or "").split(",")) - {""} or None
    z = run(fixtures, out, only, quick="--quick" in argv)
    with zipfile.ZipFile(z) as zf:
        res = json.loads(zf.read("results.json"))["results"]
    return 0 if res and all(r["status"] == "PASS" for r in res) else 1
