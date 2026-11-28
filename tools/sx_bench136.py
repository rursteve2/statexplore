"""v1.3.6 benchmark additions (corrective patch, Gate 3). Developer-only.

Runs alongside tools/sx_bench134.py (52A, 53A, 53B, 53C budget 100 + cancel, 54B) and tools/sx_bench135.py (51H3, 60F):
  FP    identity fingerprint overhead on the 54B 100,000-row file: first data_signature (cold: hashes the values in
        v1.3.6), 1,000 further data_signature + fit_identity calls (warm), and save + reopen + signature of a project
  VAL   55L AMD validation (holdout + 2-fold CV + 3 bootstrap draws, whole pipeline, 1 worker): real fits, exported and
        not reproduced (export time measured separately)
Works on every version (v1.3.4 / v1.3.5 / v1.3.6): only public functions of all three are used.

    python tools/sx_bench136.py --pkg <fixtures folder> --out <dir> [--repeats 3] [--only FP,VAL]
A baseline checkout is benchmarked with PYTHONPATH=<its src>.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sx_bench import Sampler  # noqa: E402


def fingerprints(pkg: Path, repeats: int) -> dict:

    from statexplore.core.importers import import_file
    from statexplore.core.project import Project
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings, view

    s = SearchSettings(outcome="Y", family="gaussian", participant="StudentID", time="Time", required=["Time"],
                       structure="ri", cluster="School", strategy="focal", covariate_mode="focal", max_predictors=3,
                       min_predictors=2, candidates=[f"X{i:02d}" for i in range(1, 7)], study="longitudinal")
    out = []
    for _ in range(repeats):
        d = import_file(pkg / "StatExplore_Test54B_100K.csv")
        with Sampler() as smp:
            t0 = time.perf_counter()
            engine.data_signature(view(d, s))
            cold = time.perf_counter() - t0
            t1 = time.perf_counter()
            for _ in range(1000):
                engine.data_signature(view(d, s))
                engine.fit_identity(d, s, "lmm")
            warm = time.perf_counter() - t1
            p = Project(d)
            run = engine.create_run(d, s)
            p.searches.append(run)
            tmp = Path(tempfile.mkdtemp())
            t2 = time.perf_counter()
            p.save(tmp / "p.sxp")
            q = Project.open(tmp / "p.sxp")
            same = engine.data_signature(view(q.dataset, s)) == run.signature
            reopen = time.perf_counter() - t2
        out.append({"cold_signature_s": round(cold, 3), "warm_1000_signature_and_fit_identity_s": round(warm, 3),
                    "save_reopen_signature_s": round(reopen, 3), "signature_stable_after_reopen": same,
                    "wall_s": round(cold + warm + reopen, 3), "peak_mb": round(smp.peak / 1024)})
    return {"runs": out, "median_s": statistics.median(r["wall_s"] for r in out)}


def validation_55l(pkg: Path, repeats: int) -> dict:
    from statexplore.analysis.session import RSession
    from statexplore.core.importers import import_file
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings
    from statexplore.rbridge.worker import RWorker

    d = import_file(pkg / "StatExplore_Test55L_LegacyUnderflow.csv")
    run = engine.create_run(d, SearchSettings(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian",
                                              strategy="focal", covariate_mode="focal", max_predictors=2, workers=1,
                                              max_models=None, max_minutes=None))
    engine.execute(run, d)
    out = []
    for _ in range(repeats):
        with RWorker() as w, Sampler() as smp:
            cfg = V.ValidationConfig(target="rows", modes=["holdout", "cv", "bootstrap"], k=2, B=3, seed=7,
                                     max_fits=100_000)
            t0 = time.perf_counter()
            v = V.ValidationRunner(RSession(w, Path(tempfile.mkdtemp())), d, run, cfg).execute()
            fit = time.perf_counter() - t0
            t1 = time.perf_counter()
            V.export(v, tempfile.mkdtemp(), ds=d, include_data=True, run=run)
            exp = time.perf_counter() - t1
        out.append({"wall_s": round(fit, 3), "export_s": round(exp, 3), "status": v["status"],
                    "fits_used": v["fit_budget"]["used"], "peak_mb": round(smp.peak / 1024)})
    return {"runs": out, "median_s": statistics.median(r["wall_s"] for r in out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    from statexplore import __version__

    res = {"version": __version__, "repeats": a.repeats, "workloads": {}}
    if not only or "FP" in only:
        res["workloads"]["FP"] = fingerprints(a.pkg, a.repeats)
    if not only or "VAL" in only:
        res["workloads"]["VAL"] = validation_55l(a.pkg, a.repeats)
    (a.out / "bench136.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
