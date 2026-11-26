"""StatExplore reproducible benchmark (v1.3.3, 04 §6 Test 53). Developer-only.

Workloads are synthetic or the supplied 43B fixture, with recorded seeds and declared candidate spaces; no arbitrary
time limits are asserted. Each workload runs `--repeats` times (cold = first run in a fresh R worker, warm = later);
wall time, first-result time, throughput, peak process memory (python + Rscript, sampled with Windows tasklist every
0.25 s, so it is approximate and assumes no other StatExplore / R process runs), cancellation latency and orphaned R
processes after cancellation are recorded.

    python tools/sx_bench.py --out E:\\sx_bench --repeats 3
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("PYTHONPATH"):            # a baseline checkout can be benchmarked with PYTHONPATH=<its src>
    sys.path.insert(0, str(ROOT / "src"))


def _rss_kb() -> int:
    """Sum of working sets of python.exe (this process) and every Rscript.exe (KB)."""
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0
    total = 0
    me = os.getpid()
    for row in csv.reader(out.splitlines()):
        if len(row) < 5:
            continue
        name, pid, mem = row[0].lower(), row[1], row[4]
        if name == "rscript.exe" or (name == "python.exe" and pid == str(me)):
            total += int("".join(ch for ch in mem if ch.isdigit()) or 0)
    return total


def _r_count() -> int:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Rscript.exe", "/FO", "CSV", "/NH"], capture_output=True,
                         text=True).stdout
    return sum(1 for line in out.splitlines() if "Rscript" in line)


class Sampler:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, _rss_kb())
            time.sleep(0.25)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()


def synthetic(path: Path, n_part: int, n_occ: int, n_school: int, n_pred: int, seed: int) -> None:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(1, n_part + 1), n_occ)
    school = (pid - 1) % n_school + 1
    time_ = np.tile(np.arange(n_occ), n_part)
    df = pd.DataFrame({"SCHOOL": school, "ID": pid, "TIME": time_})
    u = rng.normal(0, 1, n_part)[pid - 1] + rng.normal(0, 0.7, n_school)[school - 1]
    y = 10 + 0.4 * time_ + u
    for j in range(n_pred):
        x = rng.normal(0, 1, n_part)[pid - 1] if j % 2 else rng.integers(0, 2, n_part)[pid - 1].astype(float)
        df[f"X{j + 1}"] = x
        if j < 2:
            y = y + 0.3 * x + (0.2 * x * time_ if j == 0 else 0)
    df["Y"] = y + rng.normal(0, 1, len(df))
    df.to_csv(path, index=False)


def main() -> int:
    from statexplore import __version__
    from statexplore.core.importers import import_file
    from statexplore.discovery import engine
    from statexplore.discovery.space import SearchSettings
    try:
        from statexplore.discovery import validate as V
    except ImportError:                          # baseline without AMD validation
        V = None

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--f43", type=Path, default=None)
    a = ap.parse_args()
    out = a.out / f"bench_{dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    tmp = Path(tempfile.mkdtemp(dir=str(out)))
    med = tmp / "medium_16k.csv"
    synthetic(med, 2000, 8, 100, 10, seed=5301)                    # 16,000 rows, 100 schools, 10 predictors
    f43 = a.f43 or ROOT / "tests" / "fixtures" / "v133" / "Test43B.csv"
    loads = [
        ("small_43B_A_rs_ar1_time", f43, dict(candidates=["GROUP"], structure="rs_ar1", strategy="time")),
        ("small_43B_B_9_structures", f43, dict(candidates=["GROUP"], workflow="structure", time_interactions=True,
                                               structures=["ri", "rs_uncorr", "rs", "ar1", "diag_ar1", "rs_ar1", "car1",
                                                           "diag_car1", "rs_car1"])),
        ("medium_16k_A_ri_main_10pred_cap3", med, dict(candidates=[f"X{j}" for j in range(1, 11)], structure="ri",
                                                       strategy="main", max_predictors=3)),
        ("medium_16k_A_ar1_time_10pred", med, dict(candidates=[f"X{j}" for j in range(1, 11)], structure="ar1",
                                                   strategy="time", max_predictors=3)),
    ]
    rows = []
    for name, path, kw in loads:
        d = import_file(path)
        base = dict(outcome="Y", family="gaussian", participant="ID", time="TIME", cluster="SCHOOL", workers=4,
                    max_models=None, max_minutes=None, study="longitudinal")
        base.update(kw)
        times = []
        for rep in range(a.repeats):
            run = engine.create_run(d, SearchSettings(**base))
            first = {}
            t0 = time.perf_counter()

            def prog(r, first=first, t0=t0):
                if r.n_done and "t" not in first:
                    first["t"] = time.perf_counter() - t0
            with Sampler() as smp:
                engine.execute(run, d, progress=prog)
            wall = time.perf_counter() - t0
            times.append(wall)
            rows.append({"workload": name, "repeat": rep + 1, "rows": len(d.raw), "candidates": run.total,
                         "status": run.status, "wall_s": round(wall, 3), "first_result_s": round(first.get("t", wall), 3),
                         "candidates_per_s": round(run.total / wall, 3), "peak_rss_mb": round(smp.peak / 1024, 1)})
            print(rows[-1], flush=True)
    # validation workload (43B, cluster target, holdout + 5-fold CV + 20 bootstrap draws, pipeline)
    from statexplore.analysis.session import RSession
    from statexplore.rbridge.worker import RWorker

    if V is not None:
        d = import_file(f43)
        run = engine.create_run(d, SearchSettings(outcome="Y", candidates=["GROUP"], family="gaussian", participant="ID",
                                                  time="TIME", cluster="SCHOOL", structure="ri", strategy="time",
                                                  include_main=True, workers=1, max_models=None, max_minutes=None))
        engine.execute(run, d)
        w = RWorker()
        w.start()
        sess = RSession(w, tmp)
        for rep in range(a.repeats):
            cfg = V.ValidationConfig(target="cluster", modes=["holdout", "cv", "bootstrap"], k=5, B=20)
            t0 = time.perf_counter()
            with Sampler() as smp:
                v = V.ValidationRunner(sess, d, run, cfg).execute()
            wall = time.perf_counter() - t0
            rows.append({"workload": "validation_43B_cluster_holdout_cv5_B20_pipeline", "repeat": rep + 1, "rows": 960,
                         "candidates": len(v["candidates"]), "status": v["status"], "wall_s": round(wall, 3),
                         "first_result_s": "", "candidates_per_s": "", "peak_rss_mb": round(smp.peak / 1024, 1)})
            print(rows[-1], flush=True)
        w.stop()
    # cancellation latency + orphan check (medium search cancelled after 3 s)
    d = import_file(med)
    run = engine.create_run(d, SearchSettings(outcome="Y", candidates=[f"X{j}" for j in range(1, 11)], family="gaussian",
                                              participant="ID", time="TIME", cluster="SCHOOL", structure="rs_ar1",
                                              strategy="main", max_predictors=4, workers=4, max_models=None,
                                              max_minutes=None))
    cancel = threading.Event()
    th = threading.Thread(target=lambda: engine.execute(run, d, cancel=cancel))
    r_before = _r_count()
    th.start()
    time.sleep(3)
    t0 = time.perf_counter()
    cancel.set()
    th.join()
    latency = time.perf_counter() - t0
    time.sleep(2)
    cancel_rec = {"cancel_latency_s": round(latency, 3), "status_after_cancel": run.status, "done": run.n_done,
                  "total": run.total, "r_processes_before": r_before, "r_processes_after": _r_count()}
    print(cancel_rec, flush=True)
    summary = {"statexplore": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
               "cpu": platform.processor(), "cores": os.cpu_count(), "repeats": a.repeats,
               "medians": {n: statistics.median(r["wall_s"] for r in rows if r["workload"] == n)
                           for n in dict.fromkeys(r["workload"] for r in rows)},
               "cancellation": cancel_rec, "seeds": {"medium_16k": 5301}}
    with (out / "bench.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "bench_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
