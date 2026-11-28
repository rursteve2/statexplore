"""G0 red evidence (real R): AUD135-04 partial replay, AUD135-06 cross-sheet cache reuse, AUD135-07 historical t-test
export after a holdout seed change. Version-neutral: works on v1.3.5 and v1.3.6 (export paths resolved either way)."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import statexplore
from statexplore.core.importers import import_file
from statexplore.discovery import engine, validate as V
from statexplore.discovery.space import SearchSettings
from statexplore.analysis.session import RSession
from statexplore.rbridge.worker import RWorker
from statexplore.rbridge.locate import find_rscript, r_library

fx = Path(sys.argv[1]); out = {"version": statexplore.__version__}
env = dict(os.environ, R_LIBS=str(r_library()), R_LIBS_USER=str(r_library()))

def run_r(cwd, script):
    r = subprocess.run([str(find_rscript()), "--vanilla", script], cwd=cwd, capture_output=True, text=True, env=env, timeout=3600)
    return r.returncode, (r.stdout + r.stderr)[-1500:]

def where(files, name):
    hits = [Path(f) for f in files if Path(f).name == name]
    return hits[0].parent if hits else None

# AUD135-04
d = import_file(fx / "StatExplore_Test55L_LegacyUnderflow.csv")
run = engine.create_run(d, SearchSettings(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal",
                        covariate_mode="focal", max_predictors=2, workers=1, max_models=None, max_minutes=None))
engine.execute(run, d)
with RWorker() as w:
    s = RSession(w, Path(tempfile.mkdtemp()))
    n = {"c": 0}
    def stop():
        n["c"] += 1; return n["c"] > 1
    v = V.ValidationRunner(s, d, run, V.ValidationConfig(target="rows", modes=["holdout", "cv"], k=2, seed=11,
                                                         max_fits=100_000)).execute(should_stop=stop)
    files = V.export(v, tempfile.mkdtemp(), ds=d, include_data=True, run=run)
    f = where(files, "validation_reproduce.R") or Path(files[0]).parent
    rc, tail = run_r(f, "validation_reproduce.R")
    chk = pd.read_csv(f / "validation_check_results.csv") if (f / "validation_check_results.csv").exists() else pd.DataFrame()
    out["AUD135-04"] = {"status": v["status"], "reproduce_exit": rc,
                        "failed_checks": chk.loc[chk["status"] == "FAIL", "check"].tolist() if len(chk) else [],
                        "tail": tail[-400:]}
# AUD135-06
a = import_file(fx / "AUD135_06_two_sheets.xlsx", sheet="A"); b = import_file(fx / "AUD135_06_two_sheets.xlsx", sheet="B")
st = SearchSettings(outcome="y", candidates=["x", "z", "w"], family="gaussian", strategy="focal", max_predictors=2,
                    workers=1, max_models=None, max_minutes=None)
ra = engine.create_run(a, st); engine.execute(ra, a)
rb = engine.create_run(b, st); cache = engine.cache_from(rb, [ra], b); engine.execute(rb, b, cache=cache)
fresh = engine.create_run(b, st); engine.execute(fresh, b)
key = lambda r: {x["key"]: x["p"] for x in r.rows}
out["AUD135-06"] = {"cache_rows_admitted_from_sheet_A": len(cache), "rows_marked_reused": sum(bool(x.get("cached")) for x in rb.rows),
                    "B_p_equal_A_p": key(rb) == key(ra), "B_p_equal_fresh_B_p": all(abs(key(rb)[k] - key(fresh)[k]) < 1e-12 for k in key(fresh)),
                    "p_A": key(ra), "p_B_reported": key(rb), "p_B_fresh": key(fresh)}
# AUD135-07
from statexplore.core.dataset import HoldoutSpec
from statexplore.core.project import Project
from statexplore.analysis.builders import BY_KEY
from statexplore.analysis.framework import run_analysis
from statexplore.export.exporters import export
t = Path(tempfile.mkdtemp()); pd.DataFrame({"score": [x + 0.5 for x in range(20)]}).to_csv(t / "s.csv", index=False)
ds = import_file(t / "s.csv"); ds.set_variable_field("score", "measure", "continuous"); ds.set_holdout(HoldoutSpec(0.25, 1))
pr = Project(ds)
with RWorker() as w:
    s = RSession(w, t / "w")
    r1, a1 = run_analysis(s, ds, BY_KEY["t_one_sample"], {"dependent": ["score"]}, {})
pr.add_result(r1, a1)
m_hist = next(tb for tb in r1.tables if tb.name == "One-sample t-test").rows[0][1]
ds.set_holdout(HoldoutSpec(0.25, 2))
try:
    w1 = export(pr, [r1.id], t / "rep", {"R"}, include_history=False, include_data=True)
except TypeError:
    w1 = export(pr, [r1.id], t / "rep", {"R"}, include_history=False)
sp = next((Path(x) for x in w1 if Path(x).suffix == ".R"), None)
res = {"historical_mean": m_hist, "runnable_script_exported": sp is not None and "not_runnable" not in sp.name}
if res["runnable_script_exported"]:
    rc, tail = run_r(sp.parent, sp.name)
    j = sp.parent / f"result_{r1.id}.json"
    if j.exists():
        tab = next(x for x in json.loads(j.read_text())["tables"] if x["name"] == "One-sample t-test")["data"]
        res.update(reproduce_exit=rc, exported_bundle_mean=tab["mean"][0], matches_historical=abs(tab["mean"][0] - m_hist) < 1e-12)
    else:
        res.update(reproduce_exit=rc, tail=tail[-300:])
out["AUD135-07"] = res
print(json.dumps(out, indent=1, default=str))
