"""v1.3.5 benchmark additions (corrective patch, Gate 3). Developer-only.

Runs the v1.3.4 owner workloads (tools/sx_bench134.py: 52A fresh / reused, 53A, 53B, 53C budget + cancel, 54B) and adds
  51H3  bootstrap validation, B = 20, seed 133, new schools, whole pipeline (18 candidates): setup vs execution time
  60F   fixed-candidate validation setup on the 147,713,444-candidate space (no fits): time and peak memory of the
        candidate retrieval and feasibility (v1.3.4 scanned the suffix of the space; not run on v1.3.4 for late indices)

    python tools/sx_bench135.py --pkg <handoff Synthetic_Datasets folder> --out <dir> [--repeats 3] [--only 51H3,60F]
A baseline checkout is benchmarked with PYTHONPATH=<its src>.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("PYTHONPATH"):
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sx_bench import Sampler  # noqa: E402


def bootstrap_51h3(pkg: Path, repeats: int) -> dict:
    from statexplore.analysis.session import RSession
    from statexplore.core.importers import import_file
    from statexplore.discovery import engine
    from statexplore.discovery import validate as V
    from statexplore.discovery.space import SearchSettings
    from statexplore.rbridge.worker import RWorker

    d = import_file(pkg / "StatExplore_Test51C_ThreeWayInteraction.csv")
    s = SearchSettings(outcome="TestScore", candidates=["Sleep", "Anxiety", "Attendance"], required=["Time", "Gender"],
                       participant="StudentID", time="Time", cluster="School", structure="rs", strategy="combinations",
                       include_main=True, max_order=3, max_predictors=5, family="gaussian", workers=4, max_models=None,
                       max_minutes=None, study="longitudinal")
    run = engine.create_run(d, s)
    engine.execute(run, d)
    w = RWorker()
    w.start()
    out = []
    try:
        for i in range(repeats):
            sess = RSession(w, Path(os.environ.get("TEMP", ".")) / f"sx_bench135_{i}")
            cfg = V.ValidationConfig(target="cluster", modes=["bootstrap"], B=20, seed=133, max_fits=10**6)
            t0 = time.perf_counter()
            with Sampler() as smp:
                r = V.ValidationRunner(sess, d, run, cfg)
                t1 = time.perf_counter()
                v = r.execute()
            t2 = time.perf_counter()
            out.append({"setup_s": round(t1 - t0, 3), "execute_s": round(t2 - t1, 3), "wall_s": round(t2 - t0, 3),
                        "peak_mb": round(smp.peak / 1024), "status": v["status"],
                        "selections": [int(b["selected_index"]) for b in v["bootstrap"]],
                        "fit_budget": {k: v.get("fit_budget", {}).get(k) for k in ("used", "cache_hits", "retries",
                                                                                   "requested_bound")}})
    finally:
        w.stop()
    return out


def fixed_setup_60f(pkg: Path, repeats: int) -> dict:
    from statexplore.core.importers import import_file
    from statexplore.discovery import validate as V
    from statexplore.discovery.engine import DiscoveryRun
    from statexplore.discovery.space import SearchSettings, candidate_key, count_candidates, enumerate_candidates

    d = import_file(pkg / "StatExplore_Test53A_30Predictors.csv")
    s = SearchSettings(outcome="Y", candidates=[f"X{i:02d}" for i in range(1, 31)], required=["Time"],
                       participant="StudentID", time="Time", strategy="combinations", include_main=True,
                       max_predictors=6, study="longitudinal", family="gaussian")
    total = count_candidates(s)
    out = []
    for idx in (0, total // 2, total - 1):
        c = next(enumerate_candidates(s, start=idx))
        run = DiscoveryRun(settings=s, family="lmm", total=total, signature="x")
        run.rows = [{"idx": idx, "key": candidate_key(c), "focal": list(c.focal), "covariates": list(c.covariates),
                     "kind": c.kind, "inter": [list(t) for t in c.inter], "status": "ok"}]
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            with Sampler() as smp:
                ok = hasattr(V, "fixed_candidate") and V.fixed_candidate(run, idx) is not None
                info = V.feasibility(d, run, V.ValidationConfig(target="participant", modes=["holdout"],
                                                                pipeline="fixed", candidate=idx))
            times.append({"wall_s": round(time.perf_counter() - t0, 4), "peak_mb": round(smp.peak / 1024),
                          "resolved": bool(ok), "problems": info["problems"]})
        out.append({"index": idx, "runs": times, "median_s": statistics.median(t["wall_s"] for t in times)})
    return {"total": total, "indices": out}


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
    if not only or "51H3" in only:
        runs = bootstrap_51h3(a.pkg, a.repeats)
        w = [r["wall_s"] for r in runs]
        res["workloads"]["51H3"] = {"runs": runs, "median_s": statistics.median(w), "range_s": [min(w), max(w)]}
    if (not only or "60F" in only):
        res["workloads"]["60F"] = fixed_setup_60f(a.pkg, a.repeats)
    (a.out / "bench135.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
