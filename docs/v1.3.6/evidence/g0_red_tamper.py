"""G0 red evidence (real R) for AUD135-01/02 on the GENERATED reproduction script of the version on PYTHONPATH."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
import statexplore
from statexplore.core.importers import import_file
from statexplore.discovery import engine, validate as V
from statexplore.discovery.space import SearchSettings
from statexplore.analysis.session import RSession
from statexplore.rbridge.worker import RWorker
from statexplore.rbridge.locate import find_rscript, r_library
fx = Path(sys.argv[1]); env = dict(os.environ, R_LIBS=str(r_library()), R_LIBS_USER=str(r_library()))
d = import_file(fx / "StatExplore_Test55L_LegacyUnderflow.csv")
run = engine.create_run(d, SearchSettings(outcome="y", candidates=["x1", "x3", "x2"], family="gaussian", strategy="focal",
                        covariate_mode="focal", max_predictors=2, workers=1, max_models=None, max_minutes=None))
engine.execute(run, d)
with RWorker() as w:
    v = V.ValidationRunner(RSession(w, Path(tempfile.mkdtemp())), d, run, V.ValidationConfig(
        target="rows", modes=["holdout", "cv", "bootstrap"], k=2, B=3, seed=7, max_fits=100_000)).execute()
files = V.export(v, tempfile.mkdtemp(), ds=d, include_data=True, run=run)
base = Path(next(f for f in files if Path(f).name == "validation_reproduce.R")).parent
def rr(f):
    r = subprocess.run([str(find_rscript()), "--vanilla", "validation_reproduce.R"], cwd=f, capture_output=True, text=True, env=env)
    return r.returncode
def edit(name, col, fn):
    t = Path(tempfile.mkdtemp()) / "t"; shutil.copytree(base, t)
    df = pd.read_csv(t / name, dtype=str, keep_default_na=False); df[col] = [fn(x) for x in df[col]]
    df.to_csv(t / name, index=False); return rr(t)
out = {"version": statexplore.__version__, "clean": rr(base)}
out["stability estimate_median"] = edit("stability.csv", "estimate_median", lambda x: "100" if x else x)
out["stability freq_of_attempted"] = edit("stability.csv", "freq_of_attempted", lambda x: "0.5")
out["stability positive_sign"] = edit("stability.csv", "positive_sign", lambda x: "0.25" if x else x)
out["stability candidate_key"] = edit("stability.csv", "candidate_key", lambda x: "x")
out["bootstrap_estimands estimate_median"] = edit("bootstrap_estimands.csv", "estimate_median", lambda x: "9")
out["realized_sizes test_units"] = edit("realized_sizes.csv", "test_units", lambda x: str(int(x) + 1))
out["candidate_trace df"] = edit("candidate_trace.csv", "df", lambda x: str(float(x) + 1) if x else "1")
out["candidate_trace flags"] = edit("candidate_trace.csv", "flags", lambda x: "edited")
out["predictions participant 0-prefix"] = edit("predictions.csv", "participant", lambda x: "0" + x)
out["bootstrap_draws original_participant"] = edit("bootstrap_draws.csv", "original_participant", lambda x: "999999")
print(json.dumps(out, indent=1))
