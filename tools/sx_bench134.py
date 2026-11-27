"""v1.3.4 owner-workload benchmark (master prompt section 7). Developer-only.

Workloads (owner's v1.3.3 observations in brackets; those were measured on another machine and are not an SLA):
  52A  Null_13, pairwise focal, n 960, 4 workers: 15 fresh [2.9 s] then the same 15 reused [0.4 s]
  53A  30 predictors, pairwise focal, 435 candidates [13.7 s]
  53B  147,713,444-candidate space: count + preview (no fits) [fast, memory bounded]
  53C  same space, model budget 100 [4.9 s]; then a cancelled run (cancel 3 s after the start: latency + orphans)
  54B  100,000 rows, 10,000 participants, 100 schools, 15 candidates [43.7 s]
Each workload runs --repeats times (the first is cold); wall time, peak memory of python + Rscript (tasklist sampling,
approximate), attempted / fitted / reused counts, and for cancellation the latency and remaining Rscript processes.

    python tools/sx_bench134.py --pkg <handoff Synthetic_Datasets folder> --out <dir> [--repeats 3] [--only 52A,54B]
A baseline checkout is benchmarked with PYTHONPATH=<its src>.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sx_bench import Sampler, _r_count  # noqa: E402


def pairwise(**kw):
    from statexplore.discovery.space import SearchSettings

    base = dict(outcome="Y", family="gaussian", participant="StudentID", time="Time", required=["Time"],
                structure="ri", cluster="School", strategy="focal", covariate_mode="focal", max_predictors=3,
                min_predictors=2, include_main=False, workers=4, max_models=None, max_minutes=None,
                study="longitudinal", correction="BH", mode="C")
    base.update(kw)
    return SearchSettings(**base)


def run_search(d, s, previous=()):
    from statexplore.discovery import engine

    run = engine.create_run(d, s)
    cache = engine.cache_from(run, list(previous), d)
    t0 = time.perf_counter()
    with Sampler() as smp:
        engine.execute(run, d, cache=cache)
    wall = time.perf_counter() - t0
    return run, {"wall_s": round(wall, 3), "peak_mb": round(smp.peak / 1024), "attempted": len(run.rows),
                 "reused": sum(1 for r in run.rows if r.get("cached")), "status": run.status,
                 "sig_bh": int(engine.tests_frame(run, interpret=False)["sig_adj"].sum()) if run.rows else 0}


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
    from statexplore.core.importers import import_file
    from statexplore.discovery import engine
    from statexplore.discovery.space import count_candidates, describe_space

    res: dict = {"version": __version__, "python": platform.python_version(), "machine": platform.platform(),
                 "cpu_count": os.cpu_count(), "repeats": a.repeats, "workloads": {}}

    def keep(name):
        return not only or name in only

    if keep("52A"):
        d = import_file(a.pkg / "StatExplore_Test51F_Null_13.csv")
        out = []
        for _ in range(a.repeats):
            r1, m1 = run_search(d, pairwise(candidates=[f"X{i:02d}" for i in range(1, 7)]))
            _r2, m2 = run_search(d, pairwise(candidates=[f"X{i:02d}" for i in range(1, 7)]), previous=[r1])
            out.append({"fresh": m1, "reused": m2})
        res["workloads"]["52A"] = out
    if keep("53A"):
        d = import_file(a.pkg / "StatExplore_Test53A_30Predictors.csv")
        res["workloads"]["53A"] = [run_search(d, pairwise(candidates=[f"X{i:02d}" for i in range(1, 31)]))[1]
                                   for _ in range(a.repeats)]
    if keep("53B") or keep("53C"):
        d = import_file(a.pkg / "StatExplore_Test53A_30Predictors.csv")
        huge = dict(candidates=[f"X{i:02d}" for i in range(1, 31)], strategy="combinations", include_main=True,
                    max_predictors=6, min_predictors=1)
        if keep("53B"):
            t0 = time.perf_counter()
            with Sampler() as smp:
                n = count_candidates(pairwise(**huge))
                info = describe_space(d, pairwise(**huge))
            res["workloads"]["53B"] = {"count": n, "preview_total": info["total"], "fits_estimate": info["fits"],
                                       "wall_s": round(time.perf_counter() - t0, 3), "peak_mb": round(smp.peak / 1024)}
        if keep("53C"):
            out = []
            for _ in range(a.repeats):
                _r, m = run_search(d, pairwise(**huge, mode="D", max_models=100))
                out.append(m)
            res["workloads"]["53C_budget100"] = out
            canc = []
            for _ in range(a.repeats):
                run = engine.create_run(d, pairwise(**huge, mode="D", max_models=2000))
                ev = threading.Event()
                th = threading.Thread(target=lambda: engine.execute(run, d, cancel=ev))
                t0 = time.perf_counter()
                th.start()
                time.sleep(3.0)
                tc = time.perf_counter()
                ev.set()
                th.join()
                time.sleep(1.0)
                canc.append({"cancel_latency_s": round(time.perf_counter() - tc - 1.0, 3),
                             "total_s": round(time.perf_counter() - t0 - 1.0, 3), "status": run.status,
                             "recorded": len(run.rows), "n_done": run.n_done, "rscript_after": _r_count()})
            res["workloads"]["53C_cancel"] = canc
    if keep("54B"):
        t0 = time.perf_counter()
        d = import_file(a.pkg / "StatExplore_Test54B_100K.csv")
        load = time.perf_counter() - t0
        out = [run_search(d, pairwise(candidates=[f"X{i:02d}" for i in range(1, 7)]))[1] for _ in range(a.repeats)]
        res["workloads"]["54B"] = {"load_s": round(load, 2), "runs": out}
    for k, v in res["workloads"].items():
        runs = v if isinstance(v, list) else v.get("runs") if isinstance(v, dict) and "runs" in v else None
        if runs and isinstance(runs[0], dict) and "wall_s" in runs[0]:
            w = [r["wall_s"] for r in runs]
            res["workloads"][k] = {"runs": runs, "median_s": statistics.median(w), "range_s": [min(w), max(w)],
                                   **({"load_s": v["load_s"]} if isinstance(v, dict) and "load_s" in v else {})}
    (a.out / "bench134.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
